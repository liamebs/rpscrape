"""
utils/training/dataloader_utils.py

Reusable PyTorch Dataset for race batching with optional targets.
"""

import torch
from torch.utils.data import Dataset

class RaceDataset(Dataset):
    def __init__(self, batches, include_target=True):
        self.batches = batches
        self.include_target = include_target

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        b = self.batches[idx]
        batch = {
            "float_feats": torch.tensor(b["float_features"], dtype=torch.float32),
            "idx_feats": torch.tensor(b["embedding_indices"], dtype=torch.long),
            "comment_vecs": torch.tensor(b["comment_vector"], dtype=torch.float32),
            "spotlight_vecs": torch.tensor(b["spotlight_vector"], dtype=torch.float32),
            "mask": torch.tensor(b["mask"], dtype=torch.bool)
        }
        if self.include_target and "winner_flag" in b:
            batch["targets"] = torch.tensor(b["winner_flag"], dtype=torch.float32)
        return batch
