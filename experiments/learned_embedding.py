# src/embedding/learned_embedding.py
#
# Learned skill embeddings for student group formation.
# Drop-in alternatives to the Laplacian eigenmap: each method takes the
# marks dataframe (same input as compute_eigenmap_pandas) and returns an
# (N, dim) coordinate matrix q for the VNS optimizer.
#
# Methods:
#   eigenmap_q(df)            - spectral baseline (mirrors the published pipeline)
#   gae_q(df)                 - graph autoencoder (reconstruction objective)
#   vgae_q(df)                - variational graph autoencoder
#   joint_embedding_q(df)     - self-supervised joint-embedding (VICReg-style),
#                               augmentations encode "absolute mark level is a
#                               nuisance" (cf. Van Assel et al., NeurIPS 2025)
#
# Dependencies: numpy, scipy, autograd (pip install autograd)

import numpy as np
import autograd.numpy as anp
from autograd import grad


# ---------------- graph construction (same recipe as the published pipeline) --


def marks_matrix(df):
    X = df.drop(columns=["No."]).to_numpy().astype(float)
    return X


def correlation_graph(X, c_thresh=0.5, denom=5.0):
    corr = np.corrcoef(X)
    W = np.where(corr >= c_thresh, np.exp(corr**2 / denom), 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def normalized_adjacency(W):
    A = W + np.eye(len(W))
    d = A.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(d))
    return Dinv @ A @ Dinv


# ---------------- spectral baseline ------------------------------------------


def eigenmap_q(df, dim=3, c_thresh=0.5, denom=5.0):
    X = marks_matrix(df)
    W = correlation_graph(X, c_thresh, denom)
    d = W.sum(1)
    d[d == 0] = 1.0
    Dinv = np.diag(1.0 / np.sqrt(d))
    L_norm = np.eye(len(W)) - Dinv @ W @ Dinv
    vals, vecs = np.linalg.eigh(L_norm)
    return vecs[:, 1 : dim + 1]


# ---------------- shared encoder ---------------------------------------------


def _init_params(rng, L, hidden, dim, variational=False):
    s1 = np.sqrt(2.0 / (L + hidden))
    s2 = np.sqrt(2.0 / (hidden + dim))
    p = {
        "W1": rng.normal(0, s1, (L, hidden)),
        "W2": rng.normal(0, s2, (hidden, dim)),
    }
    if variational:
        p["W2s"] = rng.normal(0, s2, (hidden, dim))
    return p


def _encode(params, A_hat, X):
    H = anp.maximum(A_hat @ X @ params["W1"], 0.0)
    return A_hat @ H @ params["W2"]


def _adam(loss_fn, params, steps=400, lr=1e-2, seed=0):
    g = grad(loss_fn)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        gr = g(params)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * gr[k]
            v[k] = b2 * v[k] + (1 - b2) * gr[k] ** 2
            mh = m[k] / (1 - b1**t)
            vh = v[k] / (1 - b2**t)
            params[k] = params[k] - lr * mh / (np.sqrt(vh) + eps)
    return params


def _standardize(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


# ---------------- GAE / VGAE -------------------------------------------------


def gae_q(df, dim=3, hidden=16, steps=400, seed=0, c_thresh=0.5, denom=5.0):
    rng = np.random.default_rng(seed)
    X = _standardize(marks_matrix(df))
    W = correlation_graph(marks_matrix(df), c_thresh, denom)
    A_hat = normalized_adjacency(W)
    A_bin = (W > 0).astype(float)
    pos_weight = (A_bin.size - A_bin.sum()) / max(A_bin.sum(), 1.0)
    params = _init_params(rng, X.shape[1], hidden, dim)

    def loss(p):
        Z = _encode(p, A_hat, X)
        logits = Z @ Z.T
        # weighted BCE on adjacency reconstruction
        lp = anp.logaddexp(0.0, -logits)
        ln = anp.logaddexp(0.0, logits)
        return anp.mean(pos_weight * A_bin * lp + (1 - A_bin) * ln)

    params = _adam(loss, params, steps=steps)
    return np.array(_encode(params, A_hat, X))


def vgae_q(df, dim=3, hidden=16, steps=400, seed=0, c_thresh=0.5, denom=5.0,
           kl_weight=1e-3):
    rng = np.random.default_rng(seed)
    X = _standardize(marks_matrix(df))
    W = correlation_graph(marks_matrix(df), c_thresh, denom)
    A_hat = normalized_adjacency(W)
    A_bin = (W > 0).astype(float)
    pos_weight = (A_bin.size - A_bin.sum()) / max(A_bin.sum(), 1.0)
    params = _init_params(rng, X.shape[1], hidden, dim, variational=True)
    noise = rng.normal(0, 1, (X.shape[0], dim))

    def loss(p):
        H = anp.maximum(A_hat @ X @ p["W1"], 0.0)
        mu = A_hat @ H @ p["W2"]
        logvar = A_hat @ H @ p["W2s"]
        Z = mu + anp.exp(0.5 * logvar) * noise
        logits = Z @ Z.T
        lp = anp.logaddexp(0.0, -logits)
        ln = anp.logaddexp(0.0, logits)
        rec = anp.mean(pos_weight * A_bin * lp + (1 - A_bin) * ln)
        kl = -0.5 * anp.mean(1 + logvar - mu**2 - anp.exp(logvar))
        return rec + kl_weight * kl

    params = _adam(loss, params, steps=steps)
    H = np.maximum(A_hat @ X @ params["W1"], 0.0)
    return np.array(A_hat @ H @ params["W2"])


# ---------------- joint embedding (VICReg-style) -----------------------------


def _augment(X, rng, shift=1.0, scale=0.3, noise=0.2):
    """Views that vary absolute level & scale but keep the mark *profile*.

    Encodes the invariance the published pipeline hard-coded through the
    correlation kernel: a student's absolute performance level is a nuisance.
    """
    N = X.shape[0]
    a = rng.normal(0, shift, (N, 1))          # per-student additive level shift
    s = np.exp(rng.normal(0, scale, (N, 1)))  # per-student multiplicative scale
    e = rng.normal(0, noise, X.shape)         # small independent noise
    return s * X + a + e


def joint_embedding_q(df, dim=3, hidden=16, steps=400, seed=0,
                      c_thresh=0.5, denom=5.0,
                      lam_inv=25.0, lam_var=25.0, lam_cov=1.0):
    rng = np.random.default_rng(seed)
    X = _standardize(marks_matrix(df))
    W = correlation_graph(marks_matrix(df), c_thresh, denom)
    A_hat = normalized_adjacency(W)
    params = _init_params(rng, X.shape[1], hidden, dim)
    # fixed pool of augmented view pairs (keeps autograd loss deterministic)
    n_pairs = 8
    views = [( _augment(X, rng), _augment(X, rng) ) for _ in range(n_pairs)]

    def vicreg_terms(Z1, Z2):
        inv = anp.mean((Z1 - Z2) ** 2)
        out = 0.0
        for Z in (Z1, Z2):
            Zc = Z - anp.mean(Z, 0, keepdims=True)
            std = anp.sqrt(anp.mean(Zc**2, 0) + 1e-4)
            var = anp.mean(anp.maximum(0.0, 1.0 - std))
            C = (Zc.T @ Zc) / (Z.shape[0] - 1)
            offdiag = C - anp.diag(anp.diag(C))
            cov = anp.mean(offdiag**2)
            out = out + lam_var * var + lam_cov * cov
        return lam_inv * inv + out

    def loss(p):
        total = 0.0
        for V1, V2 in views:
            Z1 = _encode(p, A_hat, V1)
            Z2 = _encode(p, A_hat, V2)
            total = total + vicreg_terms(Z1, Z2)
        return total / n_pairs

    params = _adam(loss, params, steps=steps)
    return np.array(_encode(params, A_hat, X))


# ---------------- synthetic sanity test --------------------------------------


def make_synthetic_cohort(N=84, L=16, k=3, level_sd=15.0, profile_sd=8.0,
                          noise_sd=4.0, seed=0):
    """Marks = per-student absolute level (nuisance, LARGE) + group skill
    profile (signal) + noise. Returns df (like the loader) and true groups."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    groups = rng.integers(0, k, N)
    profiles = rng.normal(0, profile_sd, (k, L))
    level = rng.normal(65, level_sd, (N, 1))
    X = level + profiles[groups] + rng.normal(0, noise_sd, (N, L))
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(L)])
    df.insert(0, "No.", [str(i) for i in range(N)])
    return df, groups


def _ari(labels_a, labels_b):
    from scipy.special import comb

    n = len(labels_a)
    cats_a, cats_b = np.unique(labels_a), np.unique(labels_b)
    cont = np.array([[np.sum((labels_a == a) & (labels_b == b)) for b in cats_b]
                     for a in cats_a])
    sum_comb = sum(comb(x, 2) for x in cont.flatten())
    sum_a = sum(comb(x, 2) for x in cont.sum(1))
    sum_b = sum(comb(x, 2) for x in cont.sum(0))
    exp = sum_a * sum_b / comb(n, 2)
    mx = 0.5 * (sum_a + sum_b)
    return (sum_comb - exp) / (mx - exp)


def _kmeans(Z, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), k, replace=False)]
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        C = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j]
                      for j in range(k)])
    return lab


if __name__ == "__main__":
    df, truth = make_synthetic_cohort(seed=1)
    methods = {
        "eigenmap (baseline)": eigenmap_q,
        "GAE": gae_q,
        "VGAE": vgae_q,
        "joint-embedding": joint_embedding_q,
    }
    print(f"{'method':<22} {'ARI vs planted skills (3 seeds)':<35}")
    for name, fn in methods.items():
        scores = []
        for s in range(3):
            kwargs = {"seed": s} if name != "eigenmap (baseline)" else {}
            q = fn(df, **kwargs)
            lab = _kmeans(q, 3, seed=s)
            scores.append(_ari(truth, lab))
        print(f"{name:<22} {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
