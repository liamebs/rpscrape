"""
batch_races.py

Batching utility for transformer input. Groups races by race_id, applies dynamic padding,
and optionally truncates based on max_runners setting.

Returns batches of dictionary objects containing:
- float_features: np.array [B, R, F]
- embedding_indices: np.array [B, R, E]
- comment_vector: np.array [B, R, D]
- spotlight_vector: np.array [B, R, D]
- mask: np.array [B, R]

Inputs:
- float_cols: list of float/int feature columns
- idx_cols: list of embedding index columns (categorical)
- nlp_cols: ['comment_vector', 'spotlight_vector']
"""

import numpy as np
import pandas as pd

def batch_races(
    df: pd.DataFrame,
    float_cols: list,
    idx_cols: list,
    nlp_cols: list,
    batch_size: int = 16,
    max_runners: int = None,
    shuffle: bool = True
):
    df = df.copy()
    race_ids = df['race_id'].unique()
    if shuffle:
        np.random.shuffle(race_ids)

    batches = []
    for i in range(0, len(race_ids), batch_size):
        batch_races = race_ids[i:i+batch_size]
        df_batch = df[df['race_id'].isin(batch_races)]

        race_groups = [group for _, group in df_batch.groupby('race_id')]
        max_len = max(len(g) for g in race_groups)
        if max_runners is not None:
            max_len = min(max_len, max_runners)

        def pad_array(arr, pad_width, pad_value=0):
            return np.pad(arr, ((0, pad_width), (0, 0)), constant_values=pad_value)

        float_features = []
        embedding_indices = []
        nlp_comment = []
        nlp_spotlight = []
        mask = []

        for group in race_groups:
            group = group.sort_values(by='draw', na_position='last')  # optional stable order
            if max_runners is not None:
                group = group.head(max_runners)

            mask_row = np.ones(len(group), dtype=np.float32)

            float_arr = group[float_cols].to_numpy(dtype=np.float32)
            idx_arr = group[idx_cols].to_numpy(dtype=np.int64)
            comment_arr = np.stack(group[nlp_cols[0]].to_numpy())
            spotlight_arr = np.stack(group[nlp_cols[1]].to_numpy())

            pad_len = max_len - len(group)

            float_features.append(pad_array(float_arr, pad_len))
            embedding_indices.append(pad_array(idx_arr, pad_len))
            nlp_comment.append(pad_array(comment_arr, pad_len))
            nlp_spotlight.append(pad_array(spotlight_arr, pad_len))
            mask.append(np.pad(mask_row, (0, pad_len), constant_values=0))

        batch = {
            'float_features': np.stack(float_features),
            'embedding_indices': np.stack(embedding_indices),
            'comment_vector': np.stack(nlp_comment),
            'spotlight_vector': np.stack(nlp_spotlight),
            'mask': np.stack(mask)
        }

        batches.append(batch)

    return batches
