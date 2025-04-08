import torch.nn as nn
from .loss_ranking_utils import get_ranking_loss
from .loss_hybrid_utils import HybridLoss

def get_loss_function(args):
    if args.loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif args.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif args.loss_type == "ranking":
        return get_ranking_loss(margin=args.margin)
    elif args.loss_type == "hybrid":
        return HybridLoss(
            ce_weight=args.ce_weight,
            rank_weight=args.rank_weight,
            margin=args.margin
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")