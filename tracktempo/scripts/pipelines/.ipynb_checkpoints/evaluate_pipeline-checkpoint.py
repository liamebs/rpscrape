import sys
import os
import joblib
import torch
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.training.dataloader_utils import RaceDataset
from utils.batching.batch_races import batch_races
from modeling.transformer_model import RaceTransformer

parser = argparse.ArgumentParser(description="Evaluate Transformer Model")
parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--data", type=str, required=True, help="Path to pickled DataFrame")
parser.add_argument("--encoders", type=str, required=True, help="Path to embedding encoders .pkl")
parser.add_argument("--outdir", type=str, default="eval_logs", help="Directory to store evaluation results")
args = parser.parse_args()

def top1_accuracy(logits, winner_flags):
    pred = torch.argmax(logits, dim=1)
    true = torch.argmax(winner_flags, dim=1)
    return (pred == true).float().mean().item()

def evaluate(model, loader, loss_type):
    model.eval()
    all_preds = []
    all_trues = []
    total = 0
    top1_hits = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            try:
                if "winner_index" in batch:
                    target = torch.tensor([batch["winner_index"]], dtype=torch.long)
                else:
                    target = torch.tensor(batch["winner_flag"], dtype=torch.float32).unsqueeze(0)
                batch["targets"] = target
            except KeyError:
                print("⚠️  Skipping batch: missing winner label")
                continue

            logits = model(
                batch["float_feats"],
                batch["idx_feats"],
                batch["comment_vecs"],
                batch["spotlight_vecs"],
                batch["mask"]
            )

            if loss_type == "cross_entropy":
                logits = logits.squeeze(1)
                pred = torch.argmax(logits, dim=1)
                true = torch.argmax(batch["targets"], dim=1)
                top1_hits += (pred == true).sum().item()
                total += 1
            else:
                logits = logits.squeeze(0)
                pred = torch.argmax(logits)
                true = torch.argmax(batch["targets"].squeeze(0))
                top1_hits += int(pred == true)
                total += 1

            all_preds.append(pred.item())
            all_trues.append(true.item())

            acc = top1_hits / total
        
        if total == 0:
            print("⚠️  No valid batches to evaluate.")
            return {
                "top1_accuracy": None,
                "total_evaluated": 0,
                "all_preds": [],
                "all_trues": []
            }
            return {
                "top1_accuracy": round(acc, 4),
                "total_evaluated": total,
                "all_preds": all_preds,
                "all_trues": all_trues
            }

def main():
    print("[+] Loading data and encoders...")
    df = pd.read_pickle(args.data)
    encoders = joblib.load(args.encoders)

    float_cols = [c for c in df.columns if "_zscore" in c or "_rank" in c or c.startswith("mentions_")]
    cat_cols = list(encoders.keys())
    nlp_cols = ["comment_vector", "spotlight_vector"]
    label_col = "winner_flag"  # for all eval, predict winner_flag as binary

    batches = batch_races(df, float_cols, cat_cols, nlp_cols, label_col=label_col, exclude_non_runners=True)
    dataset = RaceDataset(batches, include_target=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print("[+] Loading model...")
    model = RaceTransformer(
        label_encoders=encoders,
        float_dim=len(float_cols),
        embedding_dim=32,
        nlp_dim=384,
        hidden_dim=64,
        nhead=4,
        num_layers=2
    )
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))

    loss_type = "hybrid" if "hybrid" in args.model else                 "ranking" if "ranking" in args.model else                 "cross_entropy" if "ce" in args.model else "bce"

    print("[+] Running evaluation...")
    metrics = evaluate(model, loader, loss_type)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    outpath = outdir / f"metrics_{loss_type}_{timestamp}.json"

    with open(outpath, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[✓] Saved metrics to {outpath}")

if __name__ == "__main__":
    main()