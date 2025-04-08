
import torch.nn as nn

def get_loss_function(cfg):
    if cfg.loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif cfg.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Unsupported loss type: {cfg.loss_type}")
