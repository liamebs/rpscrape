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
import json
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt

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
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def main():
    print("[+] Loading training data and encoders...")
    df = pd.read_pickle("data/processed/model_ready_train_2025-04-06T04-16.pkl")
    encoders = joblib.load("data/processed/embedding_encoders_2025-04-08T09-43.pkl")

    race_ids = df["race_id"].unique()
    train_ids, val_ids = train_test_split(race_ids, test_size=0.2, random_state=42)
    df_train = df[df["race_id"].isin(train_ids)].copy()
    df_val = df[df["race_id"].isin(val_ids)].copy()

    print("[+] Preparing batches...")
    float_cols = [col for col in df.columns if "_zscore" in col or "_rank" in col or col.startswith("mentions_")]
    cat_cols = list(encoders.keys())
    nlp_cols = ["comment_vector", "spotlight_vector"]
    label_col = "winner_index" if args.loss_type == "cross_entropy" else "winner_flag"

    float_cols_path = "data/processed/float_cols_2025-04-08.json"
    with open(float_cols_path, "w") as f:
        json.dump(float_cols, f)
    print(f"[+] Saved float feature columns to {float_cols_path}")

    train_batches = batch_races(df_train, float_cols, cat_cols, nlp_cols, label_col=label_col, min_runners=5)
    val_batches = batch_races(df_val, float_cols, cat_cols, nlp_cols, label_col=label_col, min_runners=5)

    train_loader = DataLoader(RaceDataset(train_batches, include_target=True), batch_size=1, shuffle=True)
    val_loader = DataLoader(RaceDataset(val_batches, include_target=True), batch_size=1, shuffle=False)

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

    checkpoint_dir = Path(args.save_dir) / f"transformer_{args.loss_type}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"
    csv_path = checkpoint_dir / "losses.csv"

    history = []
    epochs_no_improve = 0

    print("[+] Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
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
                loss = criterion(logits, target, target)
            else:
                loss = criterion(logits, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
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
                    loss = criterion(logits, target, target)
                else:
                    loss = criterion(logits, target)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f}")

        history.append((epoch, avg_loss, avg_val_loss))
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 1:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            writer.writerow([epoch, avg_loss, avg_val_loss])

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"[✓] New best checkpoint saved: {best_path} (val loss: {best_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[!] Early stopping triggered at epoch {epoch} (no improvement in {args.patience} epochs)")
                break

    # Plot training and validation loss
    epochs, train_losses, val_losses = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = checkpoint_dir / "loss_plot.png"
    plt.savefig(plot_path)
    print(f"[+] Loss plot saved to {plot_path}")

if __name__ == "__main__":
    main()