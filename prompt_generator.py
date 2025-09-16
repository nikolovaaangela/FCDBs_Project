import json
import pandas as pd
import re

from broome import nutrient_map, food_df  # we only need food_df + nutrient_map

print("Starting prompt generation...")

prompts = []

# Hardcoded dataset (grouped means version)
reference_dataset = "20_columns_cleaned_grouped_means.csv"
target_dataset = "20_columns_cleaned_grouped_means_missing20.csv"

# Build mapping: new_food_id -> description
new_food_map = dict(zip(food_df["food_id"], food_df["description_lwr"]))


# --- Similarity filter function ---
def is_similar_food(target: str, candidate: str, threshold: float = 0.5) -> bool:
    """
    Returns True if candidate is too similar to target (should exclude).
    Uses token overlap (Jaccard similarity) while ignoring numeric tokens.
    """
    target_tokens = {t for t in re.findall(r"\w+", target.lower()) if not t.isdigit()}
    candidate_tokens = {t for t in re.findall(r"\w+", candidate.lower()) if not t.isdigit()}

    if not target_tokens or not candidate_tokens:
        return False

    overlap = len(target_tokens & candidate_tokens) / len(target_tokens | candidate_tokens)
    return overlap >= threshold


# --- Step 1: Load dataset (means) ---
df_means = pd.read_csv(target_dataset)
df_ref = pd.read_csv(reference_dataset)



# First column = new food IDs
new_food_ids = df_means.iloc[:, 0].tolist()


# --- Step 2: Iterate over foods (rows) ---
for row_idx, row in df_means.iterrows():
    new_food_id = new_food_ids[row_idx]
    food_name = new_food_map.get(new_food_id, f"Unknown food {new_food_id}")
    

    # iterate over nutrients (skip first column with IDs)
    for col in df_means.columns[1:]:
        nutrient_id = int(float(col))  # nutrient IDs are column names
        nutrient_name = nutrient_map.get(nutrient_id, f"Unknown nutrient {col}")


        if pd.isna(row[col]):  # missing nutrient value
            non_missing = df_ref[col].dropna()

            # filter references
            references = [
                (fid, val)
                for fid, val in zip(df_ref.iloc[:, 0].tolist(), non_missing)
                if not is_similar_food(food_name, new_food_map.get(fid, f"Food {fid}"))
            ]

            reference_values_str = ", ".join(
                f"{new_food_map.get(fid, f'Food {fid}')}={val}"
                for fid, val in references
            )


            prompt = (
                f"How much {nutrient_name} is in {food_name}?\n"
                f"Reference {nutrient_name} mean values (grams) for other foods: {reference_values_str}.\n"
                f"Please provide only a numerical value."
            )

            prompts.append({
                "dataset": target_dataset,
                "row": row_idx,
                "col": col,
                "food": food_name,
                "nutrient": nutrient_name,
                "prompt": prompt,
                "reference_count": len(references)
            })

print(f"Generated {len(prompts)} prompts")

with open("prompts_pomos.json", "w") as f:
    json.dump(prompts, f, indent=2)
print("Prompts saved to prompts_pomos.json")
