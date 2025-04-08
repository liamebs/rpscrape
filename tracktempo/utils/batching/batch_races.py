
import numpy as np
import torch
import pandas as pd

def batch_races(df, float_cols, idx_cols, nlp_cols, exclude_non_runners=True, label_col=None, min_runners=1):
    batches = []
    grouped = df.groupby("race_id")

    for _, race in grouped:
        if exclude_non_runners:
            race = race[race["non_runner_flag"] == False]

        if len(race) < min_runners:
            continue

        race = race.sort_values("draw" if "draw" in race.columns else "name")
        R = len(race)

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
