import sys
import os
import torch
import joblib
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.batching.batch_races import batch_races
from utils.training.dataloader_utils import RaceDataset
from modeling.transformer_model import RaceTransformer
from utils.training.loss_factory import get_loss_function

parser = argparse.ArgumentParser(description="Train TrackTempo Transformer")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--loss_type", type=str, choices=["bce", "cross_entropy", "ranking", "hybrid"], default="bce", help="Loss function type")
parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save model checkpoints")
parser.add_argument("--ce_weight", type=float, default=0.7, help="Weight of CrossEntropy loss in hybrid mode")
parser.add_argument("--rank_weight", type=float, default=0.3, help="Weight of Ranking loss in hybrid mode")
parser.add_argument("--margin", type=float, default=1.0, help="Margin for ranking loss")
args = parser.parse_args()

def main():
    print("[+] Loading training data and encoders...")
    df = pd.read_pickle("data/processed/2025/03/model_ready_train.pkl")
    encoders = joblib.load("data/processed/embedding_encoders_2025-04-08T09-43.pkl")

    print("[+] Preparing batches...")
    float_cols = [col for col in df.columns if "_zscore" in col or "_rank" in col or col.startswith("mentions_")]
    cat_cols = list(encoders.keys())
    nlp_cols = ["comment_vector", "spotlight_vector"]
    label_col = "winner_index" if args.loss_type == "cross_entropy" else "winner_flag"

    batches = batch_races(df, float_cols, cat_cols, nlp_cols, label_col=label_col, min_runners=5)
    dataset = RaceDataset(batches, include_target=True)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("[+] Initializing model and optimizer...")
    model = RaceTransformer(
        label_encoders=encoders,
        float_dim=len(float_cols),
        embedding_dim=32,
        nlp_dim=384,
        hidden_dim=64,
        nhead=4,
        num_layers=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = get_loss_function(args)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    checkpoint_dir = Path(args.save_dir) / f"transformer_{args.loss_type}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Starting training...")
    best_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            logits = model(
                batch["float_feats"],
                batch["idx_feats"],
                batch["comment_vecs"],
                batch["spotlight_vecs"],
                batch["mask"]
            )

            target = batch["targets"]
            if args.loss_type in ["bce", "ranking", "hybrid"] and target.dim() == 1:
                target = target.unsqueeze(0)

            if args.loss_type == "cross_entropy":
                logits = logits.squeeze(1)
                target = target.view(-1)

            if args.loss_type == "hybrid":
                loss = criterion(logits, target, target)  # Use same binary mask for CE + Ranking
            else:
                loss = criterion(logits, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            print(f"[✓] New best checkpoint saved: {best_path} (loss: {best_loss:.4f})")

if __name__ == "__main__":
    main()