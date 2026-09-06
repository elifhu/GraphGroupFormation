# End-to-end fair, target-diversity group formation with SSL representations.
#
# Headline method for the grouping follow-up paper:
#   encoder (GCN) + soft k-way assignment head, trained jointly with
#     L = L_JE (VICReg invariance to absolute-level shifts)
#       + lam_alpha * (within-variance fraction - alpha)^2
#       + lam_size  * soft group-size balance
#       + lam_fair  * soft sensitive-attribute balance
#   then argmax + greedy repair for hard size bounds.
#
# Baseline for comparison: two-stage (JE embedding, then k-means grouping).

import numpy as np
import autograd.numpy as anp
from autograd import grad

from learned_embedding import (
    marks_matrix, correlation_graph, normalized_adjacency, _standardize,
    _augment, _init_params, _encode, _adam, make_synthetic_cohort,
    joint_embedding_q, _kmeans, _ari,
)


# ---------------- end-to-end model -------------------------------------------


def e2e_group(df, s_attr, k=12, dim=3, hidden=16, alpha=0.7, steps=500,
              seed=0, c_thresh=0.5, denom=5.0, tau=1.0,
              lam_inv=25.0, lam_var=25.0, lam_cov=1.0,
              lam_alpha=50.0, lam_size=20.0, lam_fair=20.0):
    """Returns (labels, Z). s_attr: (N, S) binary sensitive attributes."""
    rng = np.random.default_rng(seed)
    Xraw = marks_matrix(df)
    X = _standardize(Xraw)
    W = correlation_graph(Xraw, c_thresh, denom)
    A_hat = normalized_adjacency(W)
    N = X.shape[0]
    params = _init_params(rng, X.shape[1], hidden, dim)
    params["C"] = rng.normal(0, 0.5, (k, dim))
    views = [(_augment(X, rng), _augment(X, rng)) for _ in range(6)]
    s_prop = s_attr.mean(0)  # cohort proportions

    def soft_assign(Z, C):
        d2 = anp.sum((Z[:, None, :] - C[None]) ** 2, -1)
        P = anp.exp(-d2 / tau)
        return P / anp.sum(P, 1, keepdims=True)

    def vicreg(Z1, Z2):
        inv = anp.mean((Z1 - Z2) ** 2)
        out = 0.0
        for Z in (Z1, Z2):
            Zc = Z - anp.mean(Z, 0, keepdims=True)
            std = anp.sqrt(anp.mean(Zc**2, 0) + 1e-4)
            out = out + lam_var * anp.mean(anp.maximum(0.0, 1.0 - std))
            C_ = (Zc.T @ Zc) / (Z.shape[0] - 1)
            out = out + lam_cov * anp.mean((C_ - anp.diag(anp.diag(C_))) ** 2)
        return lam_inv * inv + out

    def loss(p):
        total = 0.0
        for V1, V2 in views:
            total = total + vicreg(_encode(p, A_hat, V1), _encode(p, A_hat, V2))
        total = total / len(views)

        Z = _encode(p, A_hat, X)
        P = soft_assign(Z, p["C"])                      # (N, k)
        size = anp.sum(P, 0)                            # soft sizes
        mu_g = (P.T @ Z) / size[:, None]                # group means
        mu = anp.mean(Z, 0, keepdims=True)
        var_tot = anp.mean(anp.sum((Z - mu) ** 2, 1))
        var_win = anp.sum(P * anp.sum((Z[:, None, :] - mu_g[None]) ** 2, -1)) / N
        frac = var_win / (var_tot + 1e-8)
        total = total + lam_alpha * (frac - alpha) ** 2
        total = total + lam_size * anp.mean((size / (N / k) - 1.0) ** 2)
        prop = (P.T @ s_attr) / size[:, None]           # (k, S) group props
        total = total + lam_fair * anp.mean((prop - s_prop[None]) ** 2)
        return total

    params = _adam(loss, params, steps=steps)
    Z = np.array(_encode(params, A_hat, X))
    d2 = ((Z[:, None, :] - np.array(params["C"])[None]) ** 2).sum(-1)
    labels = d2.argmin(1)
    return _repair_sizes(labels, Z, k), Z


def _repair_sizes(labels, Z, k):
    """Greedy repair to equal-size groups (moves cheapest-to-move students)."""
    N = len(labels)
    target = N // k
    labels = labels.copy()
    C = np.array([Z[labels == j].mean(0) if (labels == j).any()
                  else Z.mean(0) for j in range(k)])
    for _ in range(3 * N):
        sizes = np.bincount(labels, minlength=k)
        over = np.where(sizes > target + (1 if N % k else 0))[0]
        under = np.where(sizes < target)[0]
        if len(over) == 0 or len(under) == 0:
            break
        j = over[0]
        idx = np.where(labels == j)[0]
        costs = ((Z[idx][:, None, :] - C[under][None]) ** 2).sum(-1)
        i_loc, u_loc = np.unravel_index(costs.argmin(), costs.shape)
        labels[idx[i_loc]] = under[u_loc]
    return labels


# ---------------- two-stage baseline -----------------------------------------


def two_stage_group(df, k=12, dim=3, seed=0, **kw):
    Z = joint_embedding_q(df, dim=dim, seed=seed, **kw)
    return _kmeans(Z, k, seed=seed), Z


# ---------------- metrics ----------------------------------------------------


def grouping_metrics(labels, Z, s_attr):
    k = labels.max() + 1
    mu = Z.mean(0)
    var_tot = np.mean(np.sum((Z - mu) ** 2, 1))
    win, bet = 0.0, 0.0
    for j in range(k):
        m = labels == j
        if not m.any():
            continue
        mu_g = Z[m].mean(0)
        win += np.sum((Z[m] - mu_g) ** 2) / len(Z)
        bet += m.sum() / len(Z) * np.sum((mu_g - mu) ** 2)
    fair = np.mean([abs(s_attr[labels == j].mean(0) - s_attr.mean(0)).mean()
                    for j in range(k) if (labels == j).any()])
    sizes = np.bincount(labels, minlength=k)
    return dict(within=win, between=bet, frac=win / (win + bet + 1e-12),
                fair_dev=fair, size_spread=sizes.max() - sizes.min())


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    df, truth = make_synthetic_cohort(N=84, seed=1, level_sd=20,
                                      profile_sd=6, noise_sd=5)
    s_attr = rng.binomial(1, [0.3, 0.25], (84, 2)).astype(float)  # e.g. gender, overseas
    k, alpha = 12, 0.7

    for name, fn in [("two-stage (JE + k-means)",
                      lambda: two_stage_group(df, k=k, seed=0)),
                     ("end-to-end (alpha=0.7)",
                      lambda: e2e_group(df, s_attr, k=k, alpha=alpha, seed=0))]:
        labels, Z = fn()
        m = grouping_metrics(labels, Z, s_attr)
        print(f"== {name}")
        print(f"   within-var {m['within']:.4f}  between-var {m['between']:.4f}"
              f"  within-frac {m['frac']:.3f}")
        print(f"   fairness dev {m['fair_dev']:.3f}  size spread {m['size_spread']}")

    # alpha controllability check
    print("== alpha sweep (end-to-end): target vs achieved within-fraction")
    for a in (0.3, 0.5, 0.7, 0.9):
        labels, Z = e2e_group(df, s_attr, k=k, alpha=a, seed=0)
        m = grouping_metrics(labels, Z, s_attr)
        print(f"   alpha {a:.1f} -> achieved {m['frac']:.3f}"
              f"  (fair_dev {m['fair_dev']:.3f}, spread {m['size_spread']})")


# ---------------- free-logit end-to-end variant (current best) ---------------


def e2e_free(df, s_attr, k=12, dim=3, hidden=16, alpha=0.7, steps=500, seed=0,
             lam_inv=25., lam_var=25., lam_cov=1., lam_alpha=200., lam_size=50.,
             lam_fair=50., tau=1.0):
    rng = np.random.default_rng(seed)
    Xraw = marks_matrix(df)
    X = _standardize(Xraw)
    W = correlation_graph(Xraw)
    A_hat = normalized_adjacency(W)
    N = X.shape[0]
    p = _init_params(rng, X.shape[1], hidden, dim)
    p["Logit"] = rng.normal(0, 0.1, (N, k))
    views = [(_augment(X, rng), _augment(X, rng)) for _ in range(6)]
    s_prop = s_attr.mean(0)

    def vicreg(Z1, Z2):
        inv = anp.mean((Z1 - Z2) ** 2)
        out = 0.0
        for Z in (Z1, Z2):
            Zc = Z - anp.mean(Z, 0, keepdims=True)
            std = anp.sqrt(anp.mean(Zc**2, 0) + 1e-4)
            out = out + lam_var * anp.mean(anp.maximum(0.0, 1.0 - std))
            C_ = (Zc.T @ Zc) / (Z.shape[0] - 1)
            out = out + lam_cov * anp.mean((C_ - anp.diag(anp.diag(C_))) ** 2)
        return lam_inv * inv + out

    def loss(p):
        total = sum(vicreg(_encode(p, A_hat, V1), _encode(p, A_hat, V2))
                    for V1, V2 in views) / len(views)
        Z = _encode(p, A_hat, X)
        P = anp.exp(p["Logit"] / tau)
        P = P / anp.sum(P, 1, keepdims=True)
        size = anp.sum(P, 0)
        mu_g = (P.T @ Z) / size[:, None]
        mu = anp.mean(Z, 0, keepdims=True)
        var_tot = anp.mean(anp.sum((Z - mu) ** 2, 1))
        var_win = anp.sum(P * anp.sum((Z[:, None, :] - mu_g[None]) ** 2, -1)) / N
        frac = var_win / (var_tot + 1e-8)
        total = total + lam_alpha * (frac - alpha) ** 2
        total = total + lam_size * anp.mean((size / (N / k) - 1.0) ** 2)
        prop = (P.T @ s_attr) / size[:, None]
        total = total + lam_fair * anp.mean((prop - s_prop[None]) ** 2)
        return total

    p = _adam(loss, p, steps=steps)
    Z = np.array(_encode(p, A_hat, X))
    labels = np.array(p["Logit"]).argmax(1)
    return _repair_sizes(labels, Z, k), Z


def diversity_swap(Z, k, seed=0, iters=4000):
    """Two-stage baseline: equal-size grouping maximizing within-group spread
    (hill-climbing swaps, VNS-lite)."""
    rng = np.random.default_rng(seed)
    N = len(Z)
    labels = np.arange(N) % k
    rng.shuffle(labels)

    def within(labels):
        w = 0.0
        for j in range(k):
            m = labels == j
            if m.sum() > 1:
                w += np.sum((Z[m] - Z[m].mean(0)) ** 2)
        return w

    best = within(labels)
    for _ in range(iters):
        i, j = rng.integers(0, N, 2)
        if labels[i] == labels[j]:
            continue
        labels[i], labels[j] = labels[j], labels[i]
        w = within(labels)
        if w > best:
            best = w
        else:
            labels[i], labels[j] = labels[j], labels[i]
    return labels
