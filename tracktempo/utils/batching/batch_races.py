"""
utils/batching/batch_races.py

Create race-level batches for training or inference.
Each batch is a dict of numpy arrays [B, R, ...]
"""

import numpy as np

def batch_races(df, float_cols, idx_cols, nlp_cols, exclude_non_runners=True, label_col=None, min_runners=1):
    """
    Group dataframe into batches of races (one batch per race_id).
    Pads to max race size in each batch (dynamic batching).

    Parameters:
        df (pd.DataFrame): Full preprocessed dataset
        float_cols (list): Continuous feature column names
        idx_cols (list): Categorical embedding index column names
        nlp_cols (list): Text vector fields (e.g., comment_vector)
        exclude_non_runners (bool): Drop non-runners
        label_col (str or None): Optional target column (e.g., 'winner_flag')
        min_runners (int): Minimum runners per race to include

    Returns:
        List of batch dictionaries
    """
    batches = []
    grouped = df.groupby("race_id")

    for _, race in grouped:
        if exclude_non_runners:
            race = race[race["non_runner_flag"] == False]

        if len(race) < min_runners:
            continue

        race = race.sort_values("draw" if "draw" in race.columns else "name")
        R = len(race)

        batch_dict = {
            "float_features": np.stack(race[float_cols].values).astype(np.float32),  # [R, F]
            "embedding_indices": race[idx_cols].values.astype(np.int64),             # [R, E]
            "mask": np.ones((R,), dtype=np.int64)                                    # [R]
        }

        for col in nlp_cols:
            batch_dict[col] = np.stack(race[col].values).astype(np.float32)          # [R, D]

        if label_col and label_col in race.columns:
            batch_dict["winner_flag"] = race[label_col].values.astype(np.float32)    # [R]

        batches.append(batch_dict)

    return batches
