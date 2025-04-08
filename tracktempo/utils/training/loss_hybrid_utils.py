import torch
import torch.nn as nn
from .loss_ranking_utils import get_ranking_loss

class HybridLoss:
    def __init__(self, ce_weight=0.7, rank_weight=0.3, margin=1.0):
        self.ce_loss = nn.CrossEntropyLoss()
        self.rank_loss = get_ranking_loss(margin)
        self.ce_weight = ce_weight
        self.rank_weight = rank_weight

    def __call__(self, logits, targets, winner_mask):
        logits = logits.squeeze(1)  # ensure shape [1, R] → [R] for CE

        ce_targets = torch.argmax(winner_mask, dim=1)  # [1] class index of winner
        ce_loss_value = self.ce_loss(logits, ce_targets)

        rank_loss_value = self.rank_loss(logits.unsqueeze(0), winner_mask)

        combined = self.ce_weight * ce_loss_value + self.rank_weight * rank_loss_value
        return combined