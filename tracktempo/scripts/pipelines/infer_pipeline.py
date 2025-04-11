import sys
import torch
import joblib
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
import json
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.batching.batch_races import batch_races
from utils.training.dataloader_utils import RaceDataset
from modeling.transformer_model import RaceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_runners", type=int, default=1, help="Minimum number of runners per race")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (best.pt)")
    args = parser.parse_args()

    # === CONFIGURATION ===
    model_checkpoint = Path(args.checkpoint)
    encoders_path = "data/processed/embedding_encoders_2025-04-08T09-43.pkl"
    float_cols_path = "data/processed/float_cols_2025-04-08.json"
    inference_data_path = "data/processed/model_ready_train_2025-04-06T04-16.pkl"

    logging.info("Loading encoders and float feature list...")
    encoders = joblib.load(encoders_path)
    with open(float_cols_path, "r") as f:
        float_cols = json.load(f)

    df = pd.read_pickle(inference_data_path)
    
    # Standardize the runner column early: convert "name" or "horse" to "runner"
    if "name" in df.columns:
        df.rename(columns={"name": "runner"}, inplace=True)
    elif "horse" in df.columns:
        df.rename(columns={"horse": "runner"}, inplace=True)

    before = len(df["race_id"].unique())

    # === Filter non-runners ===
    if "non_runner_flag" in df.columns:
        df = df[df["non_runner_flag"] == False].copy()

    # Extract races with at least min_runners participants
    grouped = df.groupby("race_id")
    eligible_races = [rid for rid, group in grouped if len(group) >= args.min_runners]
    df_filtered = df[df["race_id"].isin(eligible_races)]
    after = len(df_filtered["race_id"].unique())
    logging.info(f"Races after filtering: {after} (from {before})")

    cat_cols = list(encoders.keys())
    nlp_cols = ["comment_vector", "spotlight_vector"]

    logging.info("Preparing batches for inference (labels not required)...")
    batches = batch_races(df_filtered, float_cols, cat_cols, nlp_cols, label_col=None)
    race_ids = df_filtered.groupby("race_id").size().index.tolist()
    infer_loader = DataLoader(RaceDataset(batches, include_target=False), batch_size=1)

    logging.info("Initializing model and loading checkpoint...")
    model = RaceTransformer(
        label_encoders=encoders,
        float_dim=len(float_cols),
        embedding_dim=32,
        nlp_dim=384,
        hidden_dim=64,
        nhead=4,
        num_layers=2,
    )
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    logging.info("Running inference...")
    results = {}
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            race_id = str(race_ids[i])
            if batch["float_feats"].shape[1] < args.min_runners:
                continue  # skip races with fewer runners than required
            logits = model(
                batch["float_feats"],
                batch["idx_feats"],
                batch["comment_vecs"],
                batch["spotlight_vecs"],
                batch["mask"]
            )
            preds = torch.sigmoid(logits).squeeze().tolist()
            results[race_id] = preds

    # Save predictions keyed by race_id
    output_path = Path("data/inference/predictions_" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
