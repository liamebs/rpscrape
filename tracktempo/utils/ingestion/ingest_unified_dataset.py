import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
import argparse
import os
import sys
import json

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Dynamically set project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Directory paths
RAW_DATA_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
RESULTS_CSV_PATH = Path("data") / "raw" / "2025_03_25-2025_04_01.csv"

# Import functions from your preprocessing modules
from utils.preprocessing.flatten_day_batch_pkl import run_batch_flatten
from utils.preprocessing.clean_flattened_df import clean_flattened_dataframe
from utils.preprocessing.add_embedding_indices import add_embedding_indices
from utils.preprocessing.process_text_fields import process_text_fields

def standardize_columns(df):
    if 'name' in df.columns and 'horse' not in df.columns:
        df = df.rename(columns={"name": "horse"})
    if 'dec' in df.columns:
        df = df.rename(columns={"dec": "sp"})
    return df

def merge_results(df, results_path):
    df['course_clean'] = df['course'].str.strip().str.lower()
    df['horse_clean'] = df['horse'].str.strip().str.lower()
    if 'race_date' not in df.columns:
        if 'race_datetime' in df.columns:
            df['race_date'] = pd.to_datetime(df['race_datetime']).dt.date.astype(str)
        else:
            df['race_date'] = ""
    if 'race_time' not in df.columns and 'off_time' in df.columns:
        df['race_time'] = df['off_time'].astype(str).str.strip()
    df['join_key'] = df['course_clean'] + '|' + df['horse_clean'] + '|' + df['race_date'] + '|' + df['race_time']

    results = pd.read_csv(results_path)
    results['course_clean'] = results['course'].str.strip().str.lower()
    if "horse" in results.columns:
        results['horse_clean'] = results['horse'].str.strip().str.lower()
    elif "name" in results.columns:
        results['horse_clean'] = results['name'].str.strip().str.lower()
    else:
        raise KeyError("Results CSV must have a column for the horse's name.")
    results['date'] = results['date'].astype(str).str.strip()
    results['off'] = results['off'].astype(str).str.strip()
    results['join_key'] = results['course_clean'] + '|' + results['horse_clean'] + '|' + results['date'] + '|' + results['off']

    if 'pos' in results.columns:
        results = results.rename(columns={'pos': 'position'})
    if 'position' in results.columns and 'winner_flag' not in results.columns:
        results['winner_flag'] = (results['position'] == 1).astype(int)
    if 'dec' in results.columns:
        results = results.rename(columns={'dec': 'sp'})

    merge_cols = ['join_key']
    for col in ['position', 'winner_flag', 'sp']:
        if col in results.columns:
            merge_cols.append(col)

    results_subset = results[merge_cols]
    merged = pd.merge(df, results_subset, on='join_key', how='left')

    # Add missing_result flag
    merged['missing_result'] = merged['position'].isna() & merged['sp'].isna()

    # Tag void races: all runners in a race with missing position
    void_race_ids = merged.groupby('race_id')['position'].apply(lambda x: x.isna().all())
    merged['void_race_flag'] = merged['race_id'].map(void_race_ids).fillna(False)

    return merged

def create_unified_dataset(mode="training"):
    logging.info("Running JSON flattening...")
    timestamp_str = datetime.now().strftime('%Y-%m-%dT%H-%M')
    run_output_dir = PROCESSED_DIR / timestamp_str
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Run flattening and move output files into the timestamped folder
    run_batch_flatten()
    pkl_files = sorted(PROCESSED_DIR.glob("flattened_json_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    csv_files = sorted(PROCESSED_DIR.glob("flattened_json_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkl_files:
        raise FileNotFoundError("No pickle files found in data/processed/ after flattening.")
    latest_pkl = pkl_files[0]
    latest_csv = csv_files[0] if csv_files else None
    os.rename(latest_pkl, run_output_dir / latest_pkl.name)
    if latest_csv:
        os.rename(latest_csv, run_output_dir / latest_csv.name)

    logging.info(f"Loading flattened data from {latest_pkl.name}")
    df = pd.read_pickle(run_output_dir / latest_pkl.name)

    logging.info("Cleaning data...")
    df = clean_flattened_dataframe(df)

    logging.info("Adding embedding indices...")
    df, encoders = add_embedding_indices(df)

    logging.info("Processing NLP text fields...")
    df, embeddings, regex_features = process_text_fields(df, fields=["comment", "spotlight"])

    # Save regex features
    regex_feature_path = run_output_dir / f"regex_features.json"
    with open(regex_feature_path, "w") as f:
        json.dump(regex_features, f, indent=2)
    logging.info(f"Saved regex feature list to {regex_feature_path.name}")

    df = standardize_columns(df)

    if RESULTS_CSV_PATH.exists():
        logging.info(f"Merging results from {RESULTS_CSV_PATH.name} ...")
        df = merge_results(df, RESULTS_CSV_PATH)
    else:
        logging.warning("Results CSV not found; skipping results merge.")

    # Add missing flags for previously NaN-filled float columns
    float_cols_filled = [
        'distance_f', 'class_num', 'draw', 'or', 'rpr', 'ts',
        'trainer_ovr_runs', 'trainer_ovr_wins', 'trainer_ovr_win_pct', 'trainer_ovr_profit',
        'trainer_last_14_runs', 'trainer_last_14_wins', 'trainer_last_14_win_pct', 'trainer_last_14_profit',
        'jockey_ovr_runs', 'jockey_ovr_wins', 'jockey_ovr_win_pct', 'jockey_ovr_profit',
        'jockey_last_14_runs', 'jockey_last_14_wins', 'jockey_last_14_win_pct', 'jockey_last_14_profit',
        'rpr_rank', 'or_rank', 'rpr_zscore', 'or_zscore'
    ]
    for col in float_cols_filled:
        flag_col = f"{col}_missing"
        df[flag_col] = df[col] == 0

    # Save main unified dataset
    unified_csv = run_output_dir / "unified_dataset.csv"
    unified_pkl = run_output_dir / "unified_dataset.pkl"
    df.to_csv(unified_csv, index=False)
    df.to_pickle(unified_pkl)
    logging.info(f"Unified dataset saved to {os.path.relpath(run_output_dir, start=PROJECT_ROOT)}")

    # Save embeddings per field
    for field_name, emb_array in embeddings.items():
        npz_path = run_output_dir / f"{field_name}_embeddings.npz"
        np.savez_compressed(npz_path, data=emb_array)
        logging.info(f"Saved embeddings: {os.path.relpath(npz_path, start=PROJECT_ROOT)}")

    return df, encoders, embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a unified dataset for training/inference/analysis.")
    parser.add_argument("--mode", choices=["training", "inference"], default="training",
                        help="Mode in which to run the ingestion pipeline.")
    args = parser.parse_args()

    unified_df, encoders, embeddings = create_unified_dataset(mode=args.mode)
    logging.info("Unified ingestion pipeline complete.")
