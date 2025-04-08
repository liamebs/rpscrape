
import torch

def format_target(batch, loss_type):
    if loss_type == "cross_entropy":
        return torch.tensor([batch["winner_index"]], dtype=torch.long)
    elif loss_type == "bce":
        return torch.tensor(batch["winner_flag"], dtype=torch.float32)
    else:
        raise NotImplementedError(f"Unsupported loss type: {loss_type}")
