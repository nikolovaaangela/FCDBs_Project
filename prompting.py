import requests
import json
import re
import time

url = 'http://kt-gpu5:11435/api/generate'
MODEL_NAME = "gemma3:27b"
PROMPTS_FILE = "prompts_pomos.json"
OUTPUT_FILE = "results_filtered.json"


def extract_number(text):
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else None


# --- Load prompts from JSON file ---
print("Loading prompts...")
try:
    with open(PROMPTS_FILE, "r") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")
except FileNotFoundError:
    print(f"Error: {PROMPTS_FILE} not found!")
    print("Please run generate_prompts.py first to create the prompts file.")
    exit(1)

# --- Run prompts through the model ---
print("Starting model predictions...")
results = []

for i, p in enumerate(prompts, 1):
    payload = {
        "model": MODEL_NAME,
        "prompt": p["prompt"],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=300)  # 300 oti imme edno miljarda food refa i ne ni stiga vremeto ako ne mu daame poveke vreme
        response.raise_for_status()
        data = response.json()

        model_output = data.get("response", "").strip()
        predicted_value = extract_number(model_output)

        result = {
            **p,
            "model_response": model_output,
            "predicted_value": predicted_value
        }
        results.append(result)

        print(f"[{i}/{len(prompts)}] {p['food']} - {p['nutrient']}: {predicted_value} (refs: {p['reference_count']})")

    except Exception as e:
        error_msg = str(e)
        print(f"[{i}/{len(prompts)}] ERROR on {p['food']} - {p['nutrient']}: {error_msg}")
        
        error_result = {**p, "error": error_msg}
        results.append(error_result)

    # Small delay to avoid overwhelming the server
    time.sleep(0.3)

# Save results
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nCompleted! Saved {len(results)} results to {OUTPUT_FILE}")

# Print summary
successful_predictions = len([r for r in results if "predicted_value" in r and r["predicted_value"] is not None])
errors = len([r for r in results if "error" in r])
print(f"Summary: {successful_predictions} successful predictions, {errors} errors")