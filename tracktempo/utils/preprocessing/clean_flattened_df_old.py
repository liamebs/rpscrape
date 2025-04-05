
import pandas as pd

def clean_flattened_dataframe(df):
    """Clean and optimize the flattened runner DataFrame."""

    # Drop fully-null or unused columns
    if "GoingStick" in df.columns:
        df.drop(columns=["GoingStick"], inplace=True)

    # Convert object columns that should be numeric
    numeric_obj_cols = [
        "trainer_ovr_runs", "trainer_ovr_wins", "trainer_last_14_runs", "trainer_last_14_wins",
        "jockey_ovr_runs", "jockey_ovr_wins", "jockey_last_14_runs", "jockey_last_14_wins"
    ]
    for col in numeric_obj_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse datetime fields
    if "race_datetime" in df.columns:
        df["race_datetime"] = pd.to_datetime(df["race_datetime"], errors="coerce")

    # Convert relevant object columns to categorical
    categoricals = [
        "country", "going", "sex", "type", "class_label", "headgear",
        "race_class", "venue"
    ]
    for col in categoricals:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
