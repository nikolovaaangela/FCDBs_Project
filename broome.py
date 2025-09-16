import pandas as pd
import numpy as np
import os
import json

# Load nutrient and food mappings
nutrient_df = pd.read_csv("nutrient.csv")
food_df = pd.read_csv("food.csv")

# Nutrient mapping (id → name)
nutrient_map = dict(zip(nutrient_df["id"], nutrient_df["name"]))

# Lowercase description for mapping
food_df["description_lwr"] = food_df["description"].str.lower()

# Assign new food IDs (unique per description)
food_id_map = {food: idx for idx, food in enumerate(food_df["description_lwr"].unique(), start=1)}
food_df["food_id"] = food_df["description_lwr"].map(food_id_map)

# Map old FDC IDs to new food IDs
fdc_to_newid = dict(zip(food_df["fdc_id"], food_df["food_id"]))

# Map new food IDs to descriptions (for lookup)
newid_to_description = dict(zip(food_df["food_id"], food_df["description"]))

# Map new food IDs to old FDC IDs (if needed)
newid_to_fdc = dict(zip(food_df["food_id"], food_df["fdc_id"]))


# Helper: mean ignoring zeros
def mean_ignore_zeros(x):
    nonzero = x[x != 0]
    return nonzero.mean() if len(nonzero) > 0 else 0


# Process datasets
for file_name in os.listdir():
    if file_name.endswith("_filtered.csv") and file_name not in ["nutrient.csv", "food.csv"]:
        df = pd.read_csv(file_name)

        # Assume first column is old food ID
        old_id_col = df.columns[0]

        # Replace with new IDs
        df[old_id_col] = df[old_id_col].map(fdc_to_newid)

        # Group by new food_id and apply mean_ignore_zeros to each nutrient column
        grouped = df.groupby(old_id_col).agg(mean_ignore_zeros).reset_index()

        # Save grouped dataset
        out_name = file_name.replace("_filtered.csv", "_grouped_means.csv")
        grouped.to_csv(out_name, index=False)
        print(f"Saved dataset: {out_name} (shape={grouped.shape})")

        # Simulate missing values on grouped dataset
        seed = 42
        missing_fraction = 0.2
        df_missing = grouped.copy()
        np.random.seed(seed)
        for col in df_missing.columns[1:]:
            n_missing = int(len(df_missing) * missing_fraction)
            missing_indices = np.random.choice(df_missing.index, n_missing, replace=False)
            df_missing.loc[missing_indices, col] = np.nan

        out_name_missing = out_name.replace("_grouped_means.csv", "_grouped_means_missing20.csv")
        df_missing.to_csv(out_name_missing, index=False)
        print(f"Saved dataset with missing values: {out_name_missing} (shape={df_missing.shape})")
