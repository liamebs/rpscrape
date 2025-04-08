
import sys
import torch
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.batching.batch_races import batch_races
from modeling.transformer_model import RaceTransformer

parser = argparse.ArgumentParser(description="Run inference with TrackTempo Transformer")
parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--data", type=str, required=True, help="Path to inference dataset (.pkl)")
parser.add_argument("--encoders", type=str, required=True, help="Path to embedding label encoders (.pkl)")
parser.add_argument("--outpath", type=str, default="predictions/inference_output.csv", help="Where to save runner-level predictions")
args = parser.parse_args()

def main():
    print("[+] Loading label encoders...")
    label_encoders = joblib.load(args.encoders)

    print("[+] Loading data...")
    df = pd.read_pickle(args.data)

    print("[+] Applying label encoders to categorical columns (safe mode)...")
    cat_cols = list(label_encoders.keys())
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("__MISSING__")
        known_classes = set(label_encoders[col].classes_)
        fallback = label_encoders[col].classes_[0]
        df[col] = df[col].apply(lambda x: x if x in known_classes else fallback)
        df[col] = label_encoders[col].transform(df[col])

    print("[+] Preparing batches...")
    float_cols = [
        "distance_f", "field_size", "class_num", "draw", "age", "or", "rpr", "ts", "lbs",
        "trainer_ovr_runs", "trainer_ovr_wins", "trainer_ovr_win_pct", "trainer_ovr_profit",
        "trainer_last_14_runs", "trainer_last_14_wins", "trainer_last_14_win_pct", "trainer_last_14_profit",
        "jockey_ovr_runs", "jockey_ovr_wins", "jockey_ovr_win_pct", "jockey_ovr_profit",
        "jockey_last_14_runs", "jockey_last_14_wins", "jockey_last_14_win_pct", "jockey_last_14_profit",
        "rpr_rank", "or_rank", "rpr_zscore", "or_zscore"
    ]
    float_cols += [col for col in df.columns if col.startswith("mentions_")]

    nlp_cols = ["comment_vector", "spotlight_vector"]

    batches = batch_races(
        df,
        float_cols=float_cols,
        idx_cols=cat_cols,
        nlp_cols=nlp_cols,
        label_col=None
    )

    print("[+] Loading model...")
    model = RaceTransformer(
        label_encoders=label_encoders,
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
    all_preds = []
    with torch.no_grad():
        for batch in batches:
            logits = model(
                torch.tensor(batch["float_features"], dtype=torch.float32),
                torch.tensor(batch["embedding_indices"], dtype=torch.long),
                torch.tensor(batch["comment_vector"], dtype=torch.float32),
                torch.tensor(batch["spotlight_vector"], dtype=torch.float32),
                torch.tensor(batch["mask"], dtype=torch.bool),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)

    df["predicted_win_prob"] = all_preds

    # Ensure output directory exists
    out_path = Path(args.outpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[✓] Saving predictions to {args.outpath}")
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
