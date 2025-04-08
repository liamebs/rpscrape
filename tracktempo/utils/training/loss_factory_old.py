import torch.nn as nn
from .loss_ranking_utils import get_ranking_loss

def get_loss_function(args):
    if args.loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif args.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif args.loss_type == "ranking":
        return get_ranking_loss(margin=1.0)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")