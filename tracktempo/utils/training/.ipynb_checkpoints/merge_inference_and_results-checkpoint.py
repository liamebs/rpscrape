"""
utils/training/merge_inference_and_results.py

Merge postrace results (positions) with model_ready_infer.pkl to produce model_ready_train.pkl.
"""

import pandas as pd

def merge_inference_and_results(infer_path, results_path, output_path, enable_place_flag=False):
    """
    Merge inference dataset with race results to produce training dataset.
    
    Parameters:
        infer_path (str): Path to model_ready_infer.pkl
        results_path (str): Path to cleaned results CSV or pickle
        output_path (str): Where to save the model_ready_train.pkl
        enable_place_flag (bool): Whether to include place flag (optional logic)
    """
    # Load inference dataset
    df = pd.read_pickle(infer_path)

    # Load results
    results = pd.read_csv(results_path) if results_path.endswith(".csv") else pd.read_pickle(results_path)

    # Normalize time fields (if 'off' exists)
    if "off" in results.columns:
        results["off"] = results["off"].astype(str).str.strip()

    # Normalize course and name
    results["course"] = results["course"].str.strip().str.lower()
    results["name"] = results["name"].str.strip().str.lower()
    df["course"] = df["course"].str.strip().str.lower()
    df["name"] = df["name"].str.strip().str.lower()

    # Convert positions
    results["position"] = pd.to_numeric(results["pos"], errors="coerce")
    results["winner_flag"] = (results["pos"] == "1").astype(int)

    if enable_place_flag:
        results["place_flag"] = results["position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

    # Merge keys
    merge_keys = ["course", "name", "race_datetime"]

    # Try merging
    merged = pd.merge(df, results[["course", "name", "position", "winner_flag"] + (["place_flag"] if enable_place_flag else [])],
                      on=["course", "name"], how="left")

    # Save output
    merged.to_pickle(output_path)
    print(f"✅ Merged dataset saved to {output_path}")
    print(f"📊 Merged shape: {merged.shape}")
    print(f"🔍 Missing position count: {merged['position'].isna().sum()}")
