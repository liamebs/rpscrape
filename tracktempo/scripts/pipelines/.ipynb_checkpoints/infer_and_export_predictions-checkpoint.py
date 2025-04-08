import sys
import os
import torch
import joblib
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.batching.batch_races import batch_races
from utils.training.dataloader_utils import RaceDataset
from modeling.transformer_model import RaceTransformer

parser = argparse.ArgumentParser(description="Infer with trained RaceTransformer model")
parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--data", type=str, required=True, help="Path to inference dataset (pkl)")
parser.add_argument("--encoders", type=str, required=True, help="Path to saved label encoders")
parser.add_argument("--floatcols", type=str, required=True, help="Path to float feature list (joblib)")
parser.add_argument("--outpath", type=str, required=True, help="CSV path to save predictions")
args = parser.parse_args()

def main():
    print("[+] Loading label encoders...")
    encoders = joblib.load(args.encoders)

    print("[+] Loading data...")
    df = pd.read_pickle(args.data)

    print("[+] Applying label encoders to categorical columns (safe mode)...")
    for col, encoder in encoders.items():
        df[col] = df[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0)

    print("[+] Preparing batches...")
    float_cols = joblib.load(args.floatcols)
    cat_cols = list(encoders.keys())
    nlp_cols = ["comment_vector", "spotlight_vector"]

    batches = batch_races(df, float_cols, cat_cols, nlp_cols, label_col=None)
    dataset = RaceDataset(batches, include_target=False)
    loader = DataLoader(dataset, batch_size=1)

    print("[+] Loading model...")
    model = RaceTransformer(
        label_encoders=encoders,
        float_dim=len(float_cols),
        embedding_dim=32,
        nlp_dim=384,
        hidden_dim=64,
        nhead=4,
        num_layers=2,
    )
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    model.eval()

    print("[+] Running inference...")
    all_rows = []
    for i, batch in enumerate(loader):
        with torch.no_grad():
            logits = model(
                batch["float_feats"],
                batch["idx_feats"],
                batch["comment_vecs"],
                batch["spotlight_vecs"],
                batch["mask"]
            )
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
            for j, p in enumerate(probs):
                all_rows.append({"race_id": df.iloc[i].race_id, "horse_ix": j, "probability": p})

    pd.DataFrame(all_rows).to_csv(args.outpath, index=False)
    print(f"[✓] Saved predictions to {args.outpath}")

if __name__ == "__main__":
    main()
