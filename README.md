# Deep Graph Anticlustering for Fair Student Group Formation

Forming tutorial groups that are skill-diverse inside, comparable to one
another, and demographically fair is a combinatorially hard allocation
problem. This repository implements *deep graph anticlustering*, which
learns the student skill representation and the group assignment jointly:
the diversity of the groups becomes a parameter a department sets rather
than an outcome it hopes for.

## Method

The pipeline has four stages, trained end to end where it matters.

1. **Student graph.** Students are connected on a correlation graph built
   from their assessment mark profiles, which defines the message-passing
   neighbourhoods.
2. **Skill representation.** A two-layer graph attention encoder is trained
   with a joint-embedding (VICReg) objective on perturbed views of the
   marks, so the embedding captures the shape of a profile rather than its
   absolute level.
3. **Differentiable assignment.** A logit matrix over groups is optimised
   by gradient descent jointly with the encoder, under a prescribed
   diversity level, soft group-size balance, and demographic-fairness
   penalties.
4. **Discretisation and refinement.** A capacity-constrained greedy
   assignment returns groups whose sizes differ by at most one student,
   and a swap-based local search restores the diversity target and the
   demographic balance lost in rounding.

A variance decomposition guarantees that prescribing the within-group
diversity level simultaneously fixes the between-group separation, and a
single parameter therefore controls both.

## Repository structure

```
experiments/
  deep_anticlustering.py     encoder, joint objective, discretisation, refinement
  learned_embedding.py       graph construction, augmentations, shared utilities
  run_paper_experiments.py   reproduces every number reported in the paper
  e2e_grouping.py            end-to-end grouping and evaluation metrics
main.py, src/, configs/     spectral baseline (Laplacian eigenmap with
                            variable neighbourhood search) and data tooling
analysis/                   comparison and plotting scripts
```

## Getting started

```bash
pip install -r requirements.txt
```

The core entry point takes a marks dataframe, a binary sensitive-attribute
matrix, and the number of groups:

```python
from deep_anticlustering import deep_anticluster

labels, Z = deep_anticluster(df, s_attr, k=20, alpha=0.9, encoder="gat")
```

`labels` assigns every student to a group; `Z` holds the learned skill
embeddings. `alpha` is the prescribed within-group share of the variance,
with `alpha = 1` corresponding to groups as heterogeneous as the cohort.

## Reproducing the paper

```bash
cd experiments
python run_paper_experiments.py EIE   # or EEE, or sweep
```

The script expects a local `cohorts.npy` with the cohort marks, sensitive
attributes, and reference allocation. This file is not distributed (see
below), so the script runs only in environments with approved data access.

## Data

No student data is included in this repository, and none may be added:
mark books, cohort lists, processed arrays, and run outputs are
git-ignored. The loaders expect a marks matrix with one row per student
and one column per assessment component, identified by CID only, together
with a binary sensitive-attribute matrix. Access to the underlying records
is restricted; contact the maintainers before working with any real cohort
file.
