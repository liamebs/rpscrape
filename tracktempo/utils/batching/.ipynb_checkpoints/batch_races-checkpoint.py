"""
utils/batching/batch_races.py

Create race-level batches for training or inference.
Each batch is a dict of numpy arrays [B, R, ...]
"""

import numpy as np
import torch
import pandas as pd

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

        # 🛠 Fix type coercion for embedding indices
        race[idx_cols] = race[idx_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64")

        batch_dict = {
            "float_features": torch.tensor(race[float_cols].values, dtype=torch.float32),
            "embedding_indices": torch.tensor(race[idx_cols].values, dtype=torch.int64),
            "comment_vecs": torch.tensor(np.stack(race["comment_vector"].values), dtype=torch.float32),
            "spotlight_vecs": torch.tensor(np.stack(race["spotlight_vector"].values), dtype=torch.float32),
            "mask": torch.ones(len(race), dtype=torch.bool),  # assume all real runners
        }


        for col in nlp_cols:
            batch_dict[col] = np.stack(race[col].values).astype(np.float32)          # [R, D]

        if label_col and label_col in race.columns:
            if label_col == "winner_index":
                batch_dict["winner_index"] = int(race[label_col].iloc[0])
            else:
                batch_dict["winner_flag"] = race[label_col].values.astype(np.float32)  # [R]

        batches.append(batch_dict)

    return batches
