"""
TrackTempo Transformer Training Pipeline
Trains a transformer model on batched race data.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import os
import time
import math
import torch
import joblib
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from utils.batching.batch_races import batch_races
from utils.training.dataloader_utils import RaceDataset
from torch.utils.data import DataLoader
from modeling.transformer_model import RaceTransformer

def main():
    print("=== TrackTempo Transformer Training ===")

    # === Load training data ===
    df_train = pd.read_pickle("data/processed/2025/03/model_ready_train.pkl")
    print(f"✅ Loaded dataset: {df_train.shape}")

    # === Define feature sets ===
    float_cols = [
        "distance_f", "field_size", "class_num", "draw", "age", "or", "rpr", "ts", "lbs",
        "trainer_ovr_runs", "trainer_ovr_wins", "trainer_ovr_win_pct", "trainer_ovr_profit",
        "trainer_last_14_runs", "trainer_last_14_wins", "trainer_last_14_win_pct", "trainer_last_14_profit",
        "jockey_ovr_runs", "jockey_ovr_wins", "jockey_ovr_win_pct", "jockey_ovr_profit",
        "jockey_last_14_runs", "jockey_last_14_wins", "jockey_last_14_win_pct", "jockey_last_14_profit",
        "rpr_rank", "or_rank", "rpr_zscore", "or_zscore"
    ]
    nlp_flags = [c for c in df_train.columns if c.startswith("mentions_")]
    float_cols += nlp_flags

    idx_cols = [
        "country_idx", "going_idx", "sex_idx", "type_idx",
        "class_label_idx", "headgear_idx", "race_class_idx", "venue_idx"
    ]

    nlp_cols = ["comment_vector", "spotlight_vector"]

    # === Batching ===
    batches = batch_races(
        df_train,
        float_cols=float_cols,
        idx_cols=idx_cols,
        nlp_cols=nlp_cols,
        exclude_non_runners=True,
        label_col="winner_flag",
        min_runners=5
    )
    print(f"📦 Batches created: {len(batches)}")

    train_dataset = RaceDataset(batches, include_target=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # === Load label encoders ===
    
    # === Find latest encoder ===
    enc_files = sorted(Path("data/processed").glob("embedding_encoders_*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not enc_files:
        raise FileNotFoundError("No encoder .pkl files found in data/processed/")
    encoder_path = enc_files[0]
    print(f"🧠 Using encoders from: {encoder_path.name}")
    label_encoders = joblib.load(encoder_path)


    # === Initialize model ===
    model = RaceTransformer(
        label_encoders=label_encoders,
        float_dim=len(float_cols),
        embedding_dim=32,
        nlp_dim=384,
        hidden_dim=128,
        nhead=4,
        num_layers=2
    )
    model.train()

    # === Training config ===
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 50
    max_checkpoints = 10
    save_epochs = set([1, n_epochs] + [math.ceil(i * n_epochs / max_checkpoints) for i in range(1, max_checkpoints)])

    # === Save directory ===
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    checkpoint_dir = f"checkpoints/transformer_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"💾 Saving checkpoints to: {checkpoint_dir}")
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in tqdm(train_loader, desc=f"🧠 Epoch {epoch+1}/{n_epochs}"):
            logits = model(
                batch["float_feats"],
                batch["idx_feats"],
                batch["comment_vecs"],
                batch["spotlight_vecs"],
                batch["mask"]
            )
            targets = batch["targets"]
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        duration = time.time() - start_time
        print(f"📉 Epoch {epoch+1} — Loss: {avg_loss:.4f} — ⏱️ {duration:.1f}s")

        if (epoch + 1) in save_epochs:
            ckpt_path = f"{checkpoint_dir}/epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved model to: {ckpt_path}")

if __name__ == "__main__":
    main()
