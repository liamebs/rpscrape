
import sys
import torch
import argparse
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.batching.batch_races import batch_races
from modeling.transformer_model import RaceTransformer
from utils.evaluation.eval_metrics import compute_metrics

parser = argparse.ArgumentParser(description="Evaluate trained TrackTempo Transformer")
parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--data", type=str, required=True, help="Path to inference dataset (.pkl)")
parser.add_argument("--encoders", type=str, required=True, help="Path to embedding label encoders (.pkl)")
parser.add_argument("--loss", type=str, default="bce", help="Loss function to evaluate (default: bce)")
parser.add_argument("--outdir", type=str, default="eval_logs", help="Output directory to save logs/plots")
parser.add_argument("--visualize", action="store_true", help="Enable visualizations")
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
        label_col="winner_flag"
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

    print("[+] Running evaluation...")
    metrics = compute_metrics(model, batches)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_out = outdir / "metrics.json"
    with open(metrics_out, "w") as f:
        json.dump({k: float(v) if isinstance(v, (float, int)) else str(v) for k, v in metrics.items() if k not in ["all_preds", "all_trues"]}, f, indent=2)
    print(f"[✓] Saved metrics to {metrics_out}")

    if args.visualize:
        if "all_preds" in metrics and "all_trues" in metrics:
            preds = metrics["all_preds"]
            trues = metrics["all_trues"]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist([p for p, t in zip(preds, trues) if t == 1], bins=25, alpha=0.6, label="Winners")
            ax.hist([p for p, t in zip(preds, trues) if t == 0], bins=25, alpha=0.6, label="Non-winners")
            ax.set_title("Prediction Score Distribution")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            vis_path = outdir / "prediction_distribution.png"
            plt.savefig(vis_path)
            print(f"[✓] Saved visualization to {vis_path}")
        else:
            print("⚠️  Warning: Skipping visualization — missing all_preds or all_trues in metrics.")

if __name__ == "__main__":
    main()
