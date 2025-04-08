
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
            "float_feats": b["float_features"].clone().detach().float(),
            "idx_feats": b["embedding_indices"].clone().detach().long(),
            "comment_vecs": b["comment_vecs"].clone().detach().float(),
            "spotlight_vecs": b["spotlight_vecs"].clone().detach().float(),
            "mask": b["mask"].clone().detach().bool(),
        }

        if self.include_target:
            if "winner_flag" in b:
                batch["targets"] = torch.tensor(b["winner_flag"], dtype=torch.float32)
            elif "winner_index" in b:
                batch["targets"] = torch.tensor(b["winner_index"], dtype=torch.long)

        return batch
