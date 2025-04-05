"""
testing_utils.py

Utility functions for model testing and validation during development.
"""

import pandas as pd

def get_clean_trial_race(df):
    """
    Filters the model-ready dataframe to return a single clean race
    with no non-runners or duplicated rows (excluding NLP vector columns).
    
    Parameters:
        df (pd.DataFrame): The full model-ready dataframe
    
    Returns:
        pd.DataFrame: A filtered DataFrame containing a single race
    """
    # Drop duplicate rows, ignoring unhashable columns
    safe_cols = df.columns.difference(["comment_vector", "spotlight_vector"])
    df = df.drop_duplicates(subset=safe_cols)

    # Remove non-runners
    df = df[df["non_runner_flag"] == False]

    # Select the first available race_id
    trial_race_id = df["race_id"].unique()[0]
    df = df[df["race_id"] == trial_race_id]

    print(f"🎯 Trial race_id: {trial_race_id}")
    print(f"✅ Cleaned trial race shape: {df.shape}")
    return df
