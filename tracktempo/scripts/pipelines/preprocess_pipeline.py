"""
TrackTempo Preprocessing Pipeline (timestamped)
"""

import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
import sys
from datetime import datetime
import joblib

# === Project-root aware path handling ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# === Load configuration from YAML ===
CONFIG_PATH = PROJECT_ROOT / "config" / "preprocess_config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# Extract paths from config
RAW_DATA_DIR = PROJECT_ROOT / cfg["data"]["raw_dir"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EMBED_DIR = PROJECT_ROOT / cfg["data"]["embed_dir"]
USE_SPACY = cfg["nlp"]["use_spacy"]
VECTORIZER = cfg["nlp"]["vectorizer"]
NER_FLAGS = cfg["nlp"]["ner_flags"]
SAVE_CSV = cfg["flags"]["save_csv"]
SAVE_PKL = cfg["flags"]["save_pkl"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def flatten_json_files(raw_dir):
    from utils.preprocessing.flatten_day_batch_pkl import run_batch_flatten
    run_batch_flatten()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    pkl_files = sorted(processed_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in data/processed/ after flattening.")
    latest_pkl = pkl_files[0]
    print(f"[+] Loading latest flattened data: {latest_pkl.name}")
    return pd.read_pickle(latest_pkl)

def clean_and_embed(df):
    from utils.preprocessing.clean_flattened_df import clean_flattened_dataframe
    from utils.preprocessing.add_embedding_indices import add_embedding_indices
    df = clean_flattened_dataframe(df)
    df, encoders = add_embedding_indices(df)
    return df, encoders

def vectorize_nlp(df):
    from utils.preprocessing.process_text_fields import process_text_fields
    fields = ["comment", "spotlight"]
    df, embeddings, features = process_text_fields(df, fields=fields, model_name="all-MiniLM-L6-v2", enable_regex=NER_FLAGS)
    return df, embeddings

def save_outputs(df, encoders, embeddings, out_dir):
    timestamp_str = datetime.now().strftime('%Y-%m-%dT%H-%M')
    out_path_pkl = out_dir / f"inference_dataset_{timestamp_str}.pkl"
    out_path_csv = out_dir / f"inference_dataset_{timestamp_str}.csv"
    enc_path = out_dir / f"embedding_encoders_{timestamp_str}.pkl"
    emb_path = out_dir / f"text_embeddings_{timestamp_str}.npz"

    if SAVE_PKL:
        df.to_pickle(out_path_pkl)
        print(f"[✓] Saved Pickle: {out_path_pkl.name}")
    if SAVE_CSV:
        df.to_csv(out_path_csv, index=False)
        print(f"[✓] Saved CSV:    {out_path_csv.name}")

    joblib.dump(encoders, enc_path)
    print(f"[✓] Saved Encoders: {enc_path.name}")

    np.savez(emb_path, **embeddings)
    print(f"[✓] Saved Embeddings: {emb_path.name}")

def main():
    print("=== TrackTempo Preprocessing Pipeline ===")
    flattened = flatten_json_files(RAW_DATA_DIR)
    cleaned, encoders = clean_and_embed(flattened)
    vectorized, embeddings = vectorize_nlp(cleaned)
    save_outputs(vectorized, encoders, embeddings, OUTPUT_DIR)
    print("✅ Pipeline complete.")

if __name__ == "__main__":
    main()
