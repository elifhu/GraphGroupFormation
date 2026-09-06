import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../Year1_EIE_only.xlsx"
MODEL_GROUPS_PATH = "../outputs/2026-03-24/21-40-42/group_assignments.csv"
Q_PATH = "../processed_data/eie_0.5_5.npy"

df = pd.read_excel(DATA_PATH)
df.columns = [
    " ".join(str(c).replace("\n", " ").replace("_x000D_", " ").strip().split())
    for c in df.columns
]

model_df = pd.read_csv(MODEL_GROUPS_PATH)
q = np.load(Q_PATH)

print("MODEL columns:", model_df.columns.tolist())

manual_groups = df["tutorial"].astype(str).values
model_groups = model_df["group"].values

plt.style.use("default")

def plot_structure(ax, q, groups, title):
    # all students
    ax.scatter(
        q[:, 0], q[:, 1], q[:, 2],
        color="lightgray",
        s=26,
        alpha=0.75
    )

    unique_groups = pd.unique(groups)

    # all groups same dark color -> no rainbow
    for g in unique_groups:
        mask = groups == g
        group_q = q[mask]
        if len(group_q) == 0:
            continue

        centroid = group_q.mean(axis=0)

        # centroid
        ax.scatter(
            centroid[0], centroid[1], centroid[2],
            color="black",
            s=70,
            alpha=0.95
        )

        # lines from centroid to members
        for p in group_q:
            ax.plot(
                [centroid[0], p[0]],
                [centroid[1], p[1]],
                [centroid[2], p[2]],
                color="black",
                alpha=0.12,
                linewidth=0.8
            )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_zlabel("q3")
    ax.view_init(elev=20, azim=35)
    ax.grid(False)

fig = plt.figure(figsize=(13, 6))

ax1 = fig.add_subplot(121, projection="3d")
plot_structure(ax1, q, manual_groups, "Manual Tutorial Groups")

ax2 = fig.add_subplot(122, projection="3d")
plot_structure(ax2, q, model_groups, "Optimized Groups")

plt.tight_layout()
plt.savefig("eie_manual_vs_model_clean.png", dpi=300, bbox_inches="tight")
plt.show()