import pandas as pd

# === FILE PATHS ===
master_file = "../second year Master list for Maths and Labs.xlsx"
model_file = "../outputs/2026-03-24/21-40-42/group_assignments.csv"

# === LOAD DATA ===
df = pd.read_excel(master_file)
model = pd.read_csv(model_file, header=None)

# === FORMAT MODEL OUTPUT ===
model.columns = ["idx", "group", "CID"] + [f"m{i}" for i in range(6)]

# === CLEAN TYPES ===
df["cid"] = df["cid"].astype(str).str.strip()
model["CID"] = model["CID"].astype(str).str.strip()

# === MERGE ===
merged = df.merge(model[["CID", "group"]], left_on="cid", right_on="CID", how="left")

# === ADD NEW COLUMNS ===
merged["model_tutorial_group"] = merged["group"]

# === ADD NOTE COLUMN ===
merged["model_group_note"] = ""

mask_missing = merged["model_tutorial_group"].isna()
merged.loc[mask_missing, "model_group_note"] = "Not found in Year 1 2025 marks dataset"

# === CLEAN UP ===
merged = merged.drop(columns=["CID", "group"], errors="ignore")

# === SORT (optional but nice) ===
merged = merged.sort_values(by="model_tutorial_group", na_position="last")

# === SAVE ===
output_file = "EIE_groups_with_model_suggestion.xlsx"
merged.to_excel(output_file, index=False)

print(f"Done ✅ Saved as {output_file}")