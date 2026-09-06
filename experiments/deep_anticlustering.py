# Deep anticlustering: consolidated implementation for the IAAI paper.
# GAT/GCN encoder + VICReg joint-embedding + free-logit soft assignment
# + alpha-targeted discrete polish (swap hill-climbing).

import numpy as np
import autograd.numpy as anp
from autograd import grad
from learned_embedding import (
    marks_matrix, correlation_graph, normalized_adjacency, _standardize,
    _augment, _adam)


# ---------------- encoders ---------------------------------------------------

def _init_gcn(rng, L, h, d):
    return {"W1": rng.normal(0, np.sqrt(2/(L+h)), (L, h)),
            "W2": rng.normal(0, np.sqrt(2/(h+d)), (h, d))}

def _enc_gcn(p, A_hat, X):
    H = anp.maximum(A_hat @ X @ p["W1"], 0.0)
    return A_hat @ H @ p["W2"]

def _init_gat(rng, L, h, d):
    return {"W1": rng.normal(0, np.sqrt(2/(L+h)), (L, h)),
            "a1": rng.normal(0, 0.1, (h,)), "a2": rng.normal(0, 0.1, (h,)),
            "W2": rng.normal(0, np.sqrt(2/(h+d)), (h, d)),
            "b1": rng.normal(0, 0.1, (d,)), "b2": rng.normal(0, 0.1, (d,))}

def _att(H, u, v, M):
    e = anp.outer(H @ u, anp.ones(H.shape[0])) + anp.outer(anp.ones(H.shape[0]), H @ v)
    e = anp.where(M > 0, anp.maximum(0.2*e, e), -1e9)   # leakyrelu + mask
    e = e - anp.max(e, 1, keepdims=True)
    a = anp.exp(e); a = a / anp.sum(a, 1, keepdims=True)
    return a

def _enc_gat(p, M, X):
    H = X @ p["W1"]
    H = anp.maximum(_att(H, p["a1"], p["a2"], M) @ H, 0.0)
    Z = H @ p["W2"]
    return _att(Z, p["b1"], p["b2"], M) @ Z


# ---------------- losses -----------------------------------------------------

def _vicreg(Z1, Z2, li, lv, lc):
    inv = anp.mean((Z1 - Z2) ** 2)
    out = 0.0
    for Z in (Z1, Z2):
        Zc = Z - anp.mean(Z, 0, keepdims=True)
        std = anp.sqrt(anp.mean(Zc**2, 0) + 1e-4)
        out = out + lv * anp.mean(anp.maximum(0.0, 1.0 - std))
        C = (Zc.T @ Zc) / (Z.shape[0] - 1)
        out = out + lc * anp.mean((C - anp.diag(anp.diag(C))) ** 2)
    return li * inv + out


# ---------------- deep anticlustering ---------------------------------------

def deep_anticluster(df, s_attr, k, alpha=0.9, dim=3, hidden=16, steps=500,
                     seed=0, encoder="gat", lam_inv=25., lam_var=25.,
                     lam_cov=1., lam_alpha=200., lam_size=50., lam_fair=50.,
                     polish_iters=6000):
    rng = np.random.default_rng(seed)
    Xraw = marks_matrix(df)
    X = _standardize(Xraw)
    W = correlation_graph(Xraw)
    N, L = X.shape
    M = (W > 0).astype(float) + np.eye(N)
    A_hat = normalized_adjacency(W)

    if encoder == "gat":
        p = _init_gat(rng, L, hidden, dim); enc = lambda p, V: _enc_gat(p, M, V)
    else:
        p = _init_gcn(rng, L, hidden, dim); enc = lambda p, V: _enc_gcn(p, A_hat, V)
    p["Logit"] = rng.normal(0, 0.1, (N, k))
    views = [(_augment(X, rng), _augment(X, rng)) for _ in range(6)]
    s_prop = s_attr.mean(0)

    def loss(p):
        total = sum(_vicreg(enc(p, V1), enc(p, V2), lam_inv, lam_var, lam_cov)
                    for V1, V2 in views) / len(views)
        Z = enc(p, X)
        P = anp.exp(p["Logit"]); P = P / anp.sum(P, 1, keepdims=True)
        size = anp.sum(P, 0)
        mu_g = (P.T @ Z) / size[:, None]
        mu = anp.mean(Z, 0, keepdims=True)
        Vt = anp.mean(anp.sum((Z - mu) ** 2, 1))
        Vw = anp.sum(P * anp.sum((Z[:, None, :] - mu_g[None]) ** 2, -1)) / N
        total = total + lam_alpha * (Vw / (Vt + 1e-8) - alpha) ** 2
        total = total + lam_size * anp.mean((size / (N / k) - 1.0) ** 2)
        prop = (P.T @ s_attr) / size[:, None]
        total = total + lam_fair * anp.mean((prop - s_prop[None]) ** 2)
        return total

    p = _adam(loss, p, steps=steps)
    Z = np.array(enc(p, X))
    labels = _sized_init(np.array(p["Logit"]), k)
    labels = _alpha_fair_polish(labels, Z, s_attr, alpha, rng,
                                iters=polish_iters)
    return labels, Z


def _sized_init(logits, k):
    """Greedy size-feasible assignment from logits (base/base+1 sizes)."""
    N = len(logits)
    base, rem = divmod(N, k)
    cap = np.full(k, base); cap[:rem] += 1
    order = np.argsort(-logits.max(1))
    labels = np.full(N, -1)
    counts = np.zeros(k, int)
    for i in order:
        for j in np.argsort(-logits[i]):
            if counts[j] < cap[j]:
                labels[i] = j; counts[j] += 1; break
    return labels


def _alpha_fair_polish(labels, Z, s_attr, alpha, rng, iters=6000,
                       w_alpha=1.0, w_fair=1.0):
    """Size-preserving swap hill-climb toward the alpha target + fairness."""
    N = len(Z); k = labels.max() + 1
    mu = Z.mean(0)
    Vt = np.mean(np.sum((Z - mu) ** 2, 1))
    s_prop = s_attr.mean(0)

    def score(lab):
        Vw, fair = 0.0, 0.0
        for j in range(k):
            m = lab == j
            Vw += np.sum((Z[m] - Z[m].mean(0)) ** 2)
            fair += np.mean(np.abs(s_attr[m].mean(0) - s_prop))
        frac = (Vw / N) / (Vt + 1e-12)
        return w_alpha * (frac - alpha) ** 2 + w_fair * (fair / k) ** 2

    best = score(labels)
    for _ in range(iters):
        i, j = rng.integers(0, N, 2)
        if labels[i] == labels[j]:
            continue
        labels[i], labels[j] = labels[j], labels[i]
        sc = score(labels)
        if sc < best:
            best = sc
        else:
            labels[i], labels[j] = labels[j], labels[i]
    return labels
