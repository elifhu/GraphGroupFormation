# Deep Graph Anticlustering for Fair Student Group Formation

Code for forming skill-diverse, fair tutorial groups from assessment marks.
A graph attention encoder is trained with a joint-embedding (VICReg)
objective on a student correlation graph, a differentiable soft assignment
is optimised jointly with the representation under a prescribed diversity
level, group size bounds, and fairness penalties, and a greedy
discretisation with a swap-based refinement returns the final groups.

## Contents

- `experiments/deep_anticlustering.py` — encoder, joint objective, training,
  discretisation, refinement
- `experiments/learned_embedding.py` — graph construction, augmentations,
  shared utilities
- `experiments/run_paper_experiments.py` — reproduces every number in the
  paper (`python run_paper_experiments.py <EIE|EEE|sweep>`)
- `experiments/e2e_grouping.py` — end-to-end grouping on a cohort file
- `main.py`, `src/`, `configs/` — supporting pipeline: data loaders,
  Laplacian eigenmap, variable neighbourhood search, and run statistics

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
