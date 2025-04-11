import numpy as np
import torch
import pandas as pd
import logging

def batch_races(df, float_cols, idx_cols, nlp_cols, exclude_non_runners=True, label_col=None, min_runners=1):
    batches = []
    grouped = df.groupby("race_id")

    for _, race in grouped:
        if exclude_non_runners:
            race = race[race["non_runner_flag"] == False]

        if len(race) < min_runners:
            logging.warning(f"Race {race['race_id'].iloc[0]} skipped due to insufficient runners: {len(race)}")
            continue

        # Standardize sorting: use 'draw' if exists, otherwise 'runner'
        if "draw" in race.columns:
            sort_key = "draw"
        elif "runner" in race.columns:
            sort_key = "runner"
        else:
            sort_key = None

        if sort_key:
            race = race.sort_values(sort_key)
        else:
            logging.warning("No sorting key found ('draw' or 'runner'); proceeding with unsorted data.")

        R = len(race)

        # Convert idx_cols to numeric ensuring no stray values
        race[idx_cols] = race[idx_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64")

        batch_dict = {
            "float_features": torch.tensor(race[float_cols].values, dtype=torch.float32),
            "embedding_indices": torch.tensor(race[idx_cols].values, dtype=torch.int64),
            "comment_vecs": torch.tensor(np.stack(race["comment_vector"].values), dtype=torch.float32),
            "spotlight_vecs": torch.tensor(np.stack(race["spotlight_vector"].values), dtype=torch.float32),
            "mask": torch.ones(R, dtype=torch.bool),
        }

        for col in nlp_cols:
            batch_dict[col] = np.stack(race[col].values).astype(np.float32)

        if label_col:
            if label_col == "winner_index":
                if "winner_index" in race.columns and race["winner_index"].notna().any():
                    batch_dict["winner_index"] = int(race["winner_index"].dropna().iloc[0])
                elif "winner_flag" in race.columns:
                    batch_dict["winner_index"] = int(np.argmax(race["winner_flag"].values))
            elif label_col in race.columns:
                batch_dict["winner_flag"] = race[label_col].values.astype(np.float32)

        batches.append(batch_dict)

    return batches
