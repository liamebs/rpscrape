import pandas as pd
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=str, required=True, help="Path to predictions .json file")
parser.add_argument("--output", type=str, required=True, help="Path to save enriched CSV output")
args = parser.parse_args()

logging.info("Loading predictions and race data...")
predictions_path = Path(args.predictions)
predictions = json.loads(Path(predictions_path).read_text())
race_df = pd.read_pickle("data/processed/model_ready_train_2025-04-06T04-16.pkl")

# Standardize the runner column to "runner"
if "name" in race_df.columns:
    race_df.rename(columns={"name": "runner"}, inplace=True)
elif "horse" in race_df.columns:
    race_df.rename(columns={"horse": "runner"}, inplace=True)

# Filter non-runners
race_df = race_df[race_df["non_runner_flag"] == False].copy()

# Consistent sorting: use 'draw' if available, otherwise 'runner'
if "draw" in race_df.columns:
    sort_key = "draw"
elif "runner" in race_df.columns:
    sort_key = "runner"
else:
    sort_key = None

if sort_key:
    race_df = race_df.sort_values(sort_key)
else:
    logging.warning("No sorting key found ('draw' or 'runner'); proceeding with unsorted data.")

logging.info("Building enriched prediction table...")
all_rows = []

for race_id, group in race_df.groupby("race_id"):
    group = group.copy().reset_index(drop=True)
    if str(race_id) not in predictions:
        logging.warning(f"Missing predictions for race {race_id}, skipping.")
        continue

    pred_scores = predictions[str(race_id)]
    if len(pred_scores) != len(group):
        logging.warning(f"Prediction count ({len(pred_scores)}) != horses ({len(group)}) in race {race_id}")

    group = group.iloc[:len(pred_scores)].copy()
    group["model_score"] = pred_scores[:len(group)]
    group["model_rank"] = group["model_score"].rank(ascending=False, method="first")

    all_rows.append(group[["race_id", "runner", "model_score", "model_rank"]])

logging.info("Saving enriched predictions to CSV...")
final_df = pd.concat(all_rows, ignore_index=True)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(output_path, index=False)
logging.info(f"Done: {output_path}")
