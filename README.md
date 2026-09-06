# Fair Student Group Formation

Code for forming skill-diverse, fair tutorial groups in the Department of
Electrical and Electronic Engineering, Imperial College London. The
repository contains two generations of the system.

## 1. Deployed graph-theoretic pipeline (predecessor)

Implementation of *Fair and Skill-Diverse Student Group Formation: A
Graph-Theoretic Approach* (IEEE Signal Processing Magazine). Students are
represented on a correlation graph, embedded with a Laplacian eigenmap, and
allocated by a variable neighbourhood search under fairness and size
constraints.

- `main.py` — entry point (Hydra configs in `configs/`)
- `src/heuristic/vns.py` — `VNSGroupOptimizer` and `ExactOptimizer`
- `src/eigenmap/` — Laplacian eigenmap computation
- `src/data/` — data loaders (real, circle, simulated)
- `src/analysis/` — statistics for algorithm runs

## 2. Deep graph anticlustering (IAAI-27 submission)

Joint learning of the skill representation and the group assignment:
a graph attention encoder trained with a joint-embedding (VICReg)
objective, a differentiable soft assignment, and a greedy
discretisation with a swap-based refinement.

- `experiments/deep_anticlustering.py` — encoder, joint objective, training,
  discretisation, refinement
- `experiments/learned_embedding.py` — graph construction, augmentations,
  shared utilities
- `experiments/run_paper_experiments.py` — reproduces every number in the
  paper (`python run_paper_experiments.py <EIE|EEE|sweep>`)
- `experiments/e2e_grouping.py` — end-to-end grouping on a cohort file

## Installation

```bash
pip install -r requirements.txt
```

## Data

No student data is included in this repository, and none may be added:
mark books, cohort lists, processed arrays, and run outputs are
git-ignored. The loaders expect a marks matrix (one row per student, one
column per assessment component, identified by CID only) and a binary
sensitive-attribute matrix. Access to the underlying records is restricted;
contact the maintainers before working with any real cohort file.
