"""
TrackTempo Preprocessing Pipeline
Author: You
Date: 2025-04

Transforms raw JSON + racecard data into model-ready inference data.

Usage:
    python scripts/pipelines/preprocess_pipeline.py

This script reads configuration settings from:
    config/preprocess_config.yaml

The config file specifies:
- Input directory for raw JSON data
- Output directory for processed model-ready files
- NLP options (vectorizer, NER flags)
- Flags for output format (CSV/PKL)

Folder paths are automatically resolved relative to the project root.
"""

import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path

# === Project-root aware path handling ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to 'tracktempo/'

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


# === Function Stubs ===

def flatten_json_files(raw_dir):
    """Load and flatten JSON files in the raw data directory."""
    print(f"[+] Flattening JSON files in {raw_dir}")
    # TODO: Load and normalize JSON into flat dataframe
    return pd.DataFrame()


def clean_and_embed(df):
    """Clean dtypes and embed categorical features using LabelEncoders."""
    print("[+] Cleaning and embedding data")
    # TODO: Clean data and apply label encodings (save if needed)
    return df


def vectorize_nlp(df):
    """Apply NLP vectorization to comments and flags using spaCy or TFIDF."""
    print(f"[+] Vectorizing NLP fields using {VECTORIZER} (spaCy={USE_SPACY})")
    # TODO: Apply vectorizer (e.g., TF-IDF, spaCy, BERT) + NER tagging
    return df


def save_model_ready(df, out_dir):
    """Save final model-ready dataset to output directory in .pkl and/or .csv formats."""
    print(f"[+] Saving model-ready data to {out_dir}")
    out_path_pkl = out_dir / "model_ready_infer_march-2025.pkl"
    out_path_csv = out_dir / "model_ready_infer_march-2025.csv"

    if SAVE_PKL:
        df.to_pickle(out_path_pkl)
        print(f"    └── Saved Pickle: {out_path_pkl.name}")
    if SAVE_CSV:
        df.to_csv(out_path_csv, index=False)
        print(f"    └── Saved CSV:    {out_path_csv.name}")


# === Main pipeline ===

def main():
    print("""\n=== TrackTempo Preprocessing Pipeline ===\n""")
    flattened = flatten_json_files(RAW_DATA_DIR)
    cleaned = clean_and_embed(flattened)
    vectorized = vectorize_nlp(cleaned)
    save_model_ready(vectorized, OUTPUT_DIR)
    print("""\n✅ Preprocessing complete!\n""")


if __name__ == "__main__":
    main()
