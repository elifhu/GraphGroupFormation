import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

from src.eigenmap.eigenmap import load_eigenmaps_pandas

# ==========================================
# CONFIG
# ==========================================

DATA_PATH = "../Year1_EIE_only.xlsx"
Q_PATH = "../processed_data/eie_0.5_5.npy"

c_thresh = 0.5
denom = 5

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_excel(DATA_PATH)

# clean column names
df.columns = [
    " ".join(
        str(c)
        .replace("\n", " ")
        .replace("_x000D_", " ")
        .strip()
        .split()
    )
    for c in df.columns
]

# ==========================================
# 2. MARK COLUMNS
# ==========================================

mark_cols = [
    "ELEC40002 Analysis & Design of Circuits",
    "ELEC40003 Digital Electronic & Comp Arch",
    "ELEC40004 Programming for Engineers",
    "ELEC40006 Electronics Design Project 1",
    "ELEC40009 Topics in Elec Engineering",
    "ELEC40012 Mathematics 1",
]

X = df[mark_cols].copy()

for c in mark_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

X = X.fillna(X.mean())

# ==========================================
# 3. LOAD / COMPUTE Q SPACE
# ==========================================

q = load_eigenmaps_pandas(
    [X],
    [Q_PATH],
    c_thresh,
    denom
)[0]

print("Q shape:", q.shape)

# ==========================================
# 4. GROUPS (MANUAL)
# ==========================================

groups = df["tutorial"].astype(str).values
unique_groups = np.unique(groups)

print("Number of groups:", len(unique_groups))

# ==========================================
# 5. METRICS IN Q-SPACE
# ==========================================

within_vars = []
group_means = []
within_distances = []

for g in unique_groups:
    mask = groups == g
    group_q = q[mask]

    if len(group_q) <= 1:
        continue

    # within variance
    within_vars.append(np.var(group_q, axis=0))

    # centroid
    group_means.append(np.mean(group_q, axis=0))

    # pairwise distances
    within_distances.extend(pdist(group_q))

manual_within_q = float(np.mean(within_vars))
manual_between_q = float(np.var(np.array(group_means), axis=0).mean())
manual_pairwise_q = float(np.mean(within_distances))

print("\n=== MANUAL (Q-SPACE) ===")
print("Within:", manual_within_q)
print("Between:", manual_between_q)
print("Pairwise:", manual_pairwise_q)

# ==========================================
# 6. MODEL VALUES (senin outputtan)
# ==========================================

model_within = 0.0116
model_between = 0.0001

# ==========================================
# 7. COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "metric": ["within_q", "between_q"],
    "manual": [manual_within_q, manual_between_q],
    "model": [model_within, model_between],
})

print("\n=== FINAL COMPARISON (CORRECT) ===")
print(comparison)