import sys
import os
import torch
import joblib
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.batching.batch_races import batch_races
from utils.training.dataloader_utils import RaceDataset
from modeling.transformer_model import RaceTransformer
from utils.training.loss_factory import get_loss_function

parser = argparse.ArgumentParser(description="Evaluate TrackTempo Transformer on training set")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data", type=str, default="data/processed/2025/03/model_ready_train.pkl")
parser.add_argument("--encoders", type=str, required=True)
parser.add_argument("--outdir", type=str, default="eval_logs")
parser.add_argument("--loss_type", type=str, choices=["bce", "cross_entropy", "ranking", "hybrid"], required=True)

args = parser.parse_args()

def evaluate(model, loader):
    model.eval()
    all_preds, all_trues = [], []
    top1_hits = 0
    total = 0

    for batch in tqdm(loader, desc="Evaluating"):
        logits = model(
            batch["float_feats"],
            batch["idx_feats"],
            batch["comment_vecs"],
            batch["spotlight_vecs"],
            batch["mask"],
        )

        if args.loss_type in ["cross_entropy", "hybrid"]:
            logits = logits.squeeze(1)
            if "targets" not in batch:
                continue
            pred = torch.argmax(logits, dim=1)
            true = batch["targets"].view(-1)

        elif args.loss_type == "bce":
            logits = logits.squeeze(0)
            if "targets" not in batch:
                continue
            pred = torch.argmax(logits, dim=-1)
            true = torch.argmax(batch["targets"].squeeze(0), dim=-1)

        elif args.loss_type == "ranking":
            logits = logits.squeeze(0)
            pred = torch.argmax(logits, dim=-1)
            true = batch["targets"]

        all_preds.append(pred.item())
        all_trues.append(true.item())

        if pred.item() == true.item():
            top1_hits += 1
        total += 1

    top1_acc = top1_hits / total if total > 0 else None

    return {
        "top1_accuracy": top1_acc,
        "total_evaluated": total,
        "all_preds": all_preds,
        "all_trues": all_trues,
    }

def main():
    print("[+] Loading data and encoders...")
    df = pd.read_pickle(args.data)
    encoders = joblib.load(args.encoders)

    float_cols = [col for col in df.columns if "_zscore" in col or "_rank" in col or col.startswith("mentions_")]
    cat_cols = list(encoders.keys())
    nlp_cols = ["comment_vector", "spotlight_vector"]
    label_col = "winner_index" if args.loss_type in ["cross_entropy", "ranking", "hybrid"] else "winner_flag"

    batches = batch_races(df, float_cols, cat_cols, nlp_cols, label_col=label_col, min_runners=5)
    dataset = RaceDataset(batches, include_target=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

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

    print("[+] Running evaluation...")
    metrics = evaluate(model, loader)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = Path(args.outdir) / f"metrics_trainset_{args.loss_type}_{timestamp}.json"
    Path(args.outdir).mkdir(exist_ok=True, parents=True)

    with open(out_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    print(f"[✓] Saved training set evaluation metrics to {out_path}")

if __name__ == "__main__":
    main()
