"""
utils/training/merge_inference_and_results.py

Merge postrace results (positions) with model_ready_infer.pkl to produce model_ready_train.pkl.
"""

import pandas as pd

# 🗺️ Expanded course normalization map
COURSE_MAP = {
    "turffontein standside (saf)": "turffontein (saf)",
    "sha tin": "sha tin (hk)",
    "happy valley": "happy valley (hk)",
    "greyville polytrack (saf)": "greyville (saf)",
    "greyville turf (saf)": "greyville (saf)",
    "dundalk (aw) (ire)": "dundalk (aw)"
}

def normalize_course(course):
    course = course.strip().lower()
    return COURSE_MAP.get(course, course)

def strip_suffix(name):
    return pd.Series(name).str.strip().str.lower().str.replace(r"\s\([a-z]{2,3}\)$", "", regex=True)

def merge_inference_and_results(infer_path, results_path, output_path, enable_place_flag=False):
    """
    Merge inference dataset with race results to produce training dataset.
    
    Parameters:
        infer_path (str): Path to model_ready_infer.pkl
        results_path (str): Path to cleaned results CSV or pickle
        output_path (str): Where to save the model_ready_train.pkl
        enable_place_flag (bool): Whether to include place flag (optional logic)
    """
    # Load data
    df = pd.read_pickle(infer_path)
    results = pd.read_csv(results_path) if results_path.endswith(".csv") else pd.read_pickle(results_path)

    # Standardize inference
    df["course"] = df["course"].apply(normalize_course)
    df["name"] = strip_suffix(df["name"])
    df["race_date"] = pd.to_datetime(df["race_datetime"]).dt.date.astype(str)
    df["race_time"] = df["off_time"].astype(str).str.strip()

    # Standardize results
    results["course"] = results["course"].apply(normalize_course)
    results["horse"] = strip_suffix(results["horse"])
    results["date"] = results["date"].astype(str).str.strip()
    results["off"] = results["off"].astype(str).str.strip()

    # Position flags
    results["position"] = pd.to_numeric(results["pos"], errors="coerce")
    results["winner_flag"] = (results["pos"] == "1").astype(int)
    if enable_place_flag:
        results["place_flag"] = results["position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

    # Merge
    merged = pd.merge(
        df,
        results[["course", "horse", "date", "off", "position", "winner_flag"] + (["place_flag"] if enable_place_flag else [])],
        left_on=["course", "name", "race_date", "race_time"],
        right_on=["course", "horse", "date", "off"],
        how="left"
    )

    # Clean up
    merged.drop(columns=["horse", "date", "off"], errors="ignore", inplace=True)
    merged.to_pickle(output_path)

    print(f"✅ Merged dataset saved to {output_path}")
    print(f"📊 Merged shape: {merged.shape}")
    print(f"🔍 Missing position count: {merged['position'].isna().sum()}")
