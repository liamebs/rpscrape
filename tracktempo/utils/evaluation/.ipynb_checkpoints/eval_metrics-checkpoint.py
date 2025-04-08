import torch
import numpy as np
from sklearn.metrics import log_loss, accuracy_score

def compute_metrics(model, batches):
    all_preds = []
    all_trues = []
    has_labels = "winner_flag" in batches[0]

    with torch.no_grad():
        for batch in batches:
            logits = model(
                torch.tensor(batch["float_features"], dtype=torch.float32),
                torch.tensor(batch["embedding_indices"], dtype=torch.long),
                torch.tensor(batch["comment_vector"], dtype=torch.float32),
                torch.tensor(batch["spotlight_vector"], dtype=torch.float32),
                torch.tensor(batch["mask"], dtype=torch.bool),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)

            if has_labels:
                labels = batch["winner_flag"]
                all_trues.extend(labels)

    metrics = {
        "all_preds": np.array(all_preds),
    }

    if has_labels:
        all_trues = np.array(all_trues)
        metrics["log_loss"] = log_loss(all_trues, metrics["all_preds"], labels=[0, 1])
        metrics["accuracy"] = accuracy_score(all_trues >= 0.5, metrics["all_preds"] >= 0.5)

    return metrics
