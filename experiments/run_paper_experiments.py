"""Canonical experiment script for the deep anticlustering paper (IAAI-27).

Produces every number reported in the paper:
  - five-configuration comparison on the EIE and EEE cohorts
    (manual, eigenmap+search, JE+search, DA-GCN, DA-GAT), 3 seeds,
    evaluated in the model-independent profile space
  - alpha controllability sweep (DA-GAT, EIE)
Usage: python run_paper_experiments.py <cohort: EIE|EEE|sweep>
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

from deep_anticlustering import deep_anticluster
from e2e_grouping import grouping_metrics, diversity_swap
import learned_embedding as le

SEEDS = (0, 1, 2)
ALPHA = 0.9


def eval_space(X):
    P = X - X.mean(1, keepdims=True)
    return (P - P.mean(0)) / (P.std(0) + 1e-9)


def cohort_frame(X):
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(X.shape[1])])
    df.insert(0, "No.", [str(i) for i in range(len(X))])
    return df


def run_cohort(name):
    cohorts = np.load('cohorts.npy', allow_pickle=True).item()
    c = cohorts[name]
    X, s, manual = c['X'], c['s'], c['manual']
    k = len(np.unique(manual))
    E = eval_space(X)
    df = cohort_frame(X)
    Zeig = le.eigenmap_q(df)

    def agg(runs):
        keys = ['frac', 'fair_dev', 'size_spread', 'anova_p']
        return {kk: (np.mean([r[kk] for r in runs]),
                     np.std([r[kk] for r in runs])) for kk in keys}

    def metrics(l):
        r = grouping_metrics(l, E, s)
        means = X.mean(1)
        groups = [means[l == j] for j in range(k) if (l == j).sum() > 1]
        r['anova_p'] = stats.f_oneway(*groups).pvalue
        return r

    res = {'manual': agg([metrics(manual)])}
    res['eigenmap+search'] = agg([metrics(diversity_swap(Zeig, k, seed=sd))
                                  for sd in SEEDS])
    res['JE+search'] = agg([metrics(diversity_swap(
        le.joint_embedding_q(df, seed=sd), k, seed=sd)) for sd in SEEDS])
    for enc in ('gcn', 'gat'):
        runs = []
        for sd in SEEDS:
            l, _ = deep_anticluster(df, s, k=k, alpha=ALPHA, seed=sd,
                                    encoder=enc, steps=400)
            runs.append(metrics(l))
        res[f'DA-{enc.upper()}'] = agg(runs)

    print(f"== {name} (N={len(X)}, k={k}, alpha={ALPHA}, seeds={SEEDS})")
    for m, v in res.items():
        print(f"  {m:<16} frac {v['frac'][0]:.3f}+-{v['frac'][1]:.3f}  "
              f"fair {v['fair_dev'][0]:.3f}+-{v['fair_dev'][1]:.3f}  "
              f"spread {v['size_spread'][0]:.1f}  "
              f"anova_p {v['anova_p'][0]:.2f}")
    np.save(f'canon_{name}.npy', res, allow_pickle=True)


def run_sweep():
    cohorts = np.load('cohorts.npy', allow_pickle=True).item()
    c = cohorts['EIE']
    X, s, manual = c['X'], c['s'], c['manual']
    k = len(np.unique(manual))
    df = cohort_frame(X)
    print("== alpha sweep (DA-GAT, EIE, own skill space, seeds 0-2)")
    out = {}
    for a in (0.3, 0.5, 0.7, 0.9):
        fr = []
        for sd in SEEDS:
            l, Z = deep_anticluster(df, s, k=k, alpha=a, seed=sd,
                                    encoder="gat", steps=400)
            fr.append(grouping_metrics(l, Z, s)['frac'])
        out[a] = (np.mean(fr), np.std(fr))
        print(f"  alpha {a:.1f} -> {np.mean(fr):.3f}+-{np.std(fr):.3f}")
    np.save('canon_sweep.npy', out, allow_pickle=True)


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == 'sweep':
        run_sweep()
    else:
        run_cohort(arg)
