"""
TrackTempo Preprocessing Pipeline
Transforms raw JSON + racecard data into model-ready inference data.
"""

import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
import sys

# === Project-root aware path handling ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to 'tracktempo/'

# 💡 Append the root to Python path for clean imports
sys.path.append(str(PROJECT_ROOT))

# === Load configuration from YAML ===
CONFIG_PATH = PROJECT_ROOT / "config" / "preprocess_config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# Extract paths from config
RAW_DATA_DIR = PROJECT_ROOT / cfg["data"]["raw_dir"]
OUTPUT_DIR = PROJECT_ROOT / cfg["data"]["output_dir"]
EMBED_DIR = PROJECT_ROOT / cfg["data"]["embed_dir"]

# NLP and output format options
USE_SPACY = cfg["nlp"]["use_spacy"]
VECTORIZER = cfg["nlp"]["vectorizer"]
NER_FLAGS = cfg["nlp"]["ner_flags"]
SAVE_CSV = cfg["flags"]["save_csv"]
SAVE_PKL = cfg["flags"]["save_pkl"]

# Ensure output folder exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Actual Logic Functions ===

def flatten_json_files(raw_dir):
    from flattening.flatten_day_batch_pkl import run_batch_flatten
    print("[+] Running batch flattening script...")
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

    print("[+] Cleaning flattened data...")
    df = clean_flattened_dataframe(df)

    print("[+] Adding embedding indices...")
    df, encoders = add_embedding_indices(df)

    return df


def vectorize_nlp(df):
    from utils.preprocessing.process_text_fields import process_text_fields

    nlp_fields = ["comment", "spotlight"]
    print("[+] Running text vectorization and NER feature extraction...")
    df, embeddings, features = process_text_fields(
        df,
        fields=nlp_fields,
        model_name='all-MiniLM-L6-v2',
        enable_regex=NER_FLAGS
    )
    return df


def save_model_ready(df, out_dir):
    print(f"[+] Saving model-ready data to {out_dir}")
    out_path_pkl = out_dir / "model_ready_infer_march-2025.pkl"
    out_path_csv = out_dir / "model_ready_infer_march-2025.csv"
    if SAVE_PKL:
        df.to_pickle(out_path_pkl)
        print(f"    └── Saved Pickle: {out_path_pkl.name}")
    if SAVE_CSV:
        df.to_csv(out_path_csv, index=False)
        print(f"    └── Saved CSV:    {out_path_csv.name}")


# === Main Pipeline ===

def main():
    print("""\n=== TrackTempo Preprocessing Pipeline ===\n""")
    flattened = flatten_json_files(RAW_DATA_DIR)
    cleaned = clean_and_embed(flattened)
    vectorized = vectorize_nlp(cleaned)
    save_model_ready(vectorized, OUTPUT_DIR)
    print("""\n✅ Preprocessing complete!\n""")


if __name__ == "__main__":
    main()
