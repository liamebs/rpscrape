import torch
import torch.nn.functional as F

def get_ranking_loss(margin=1.0):
    def loss_fn(logits, winner_mask):
        # logits: [1, R] or [R]
        # winner_mask: [1, R] or [R] — binary 1 for winner(s), 0 otherwise

        if logits.dim() == 3:
            logits = logits.squeeze(0)
        if winner_mask.dim() == 3:
            winner_mask = winner_mask.squeeze(0)

        # Ensure shape [R]
        logits = logits.view(-1)
        winner_mask = winner_mask.view(-1)

        winner_scores = logits[winner_mask == 1]    # e.g. [3.7]
        loser_scores = logits[winner_mask == 0]     # e.g. [2.1, 1.8, ...]

        if len(winner_scores) == 0 or len(loser_scores) == 0:
            # Return zero loss if no valid pairs
            return torch.tensor(0.0, requires_grad=True)

        losses = []
        for w_score in winner_scores:
            loss_per_winner = torch.clamp(margin - (w_score - loser_scores), min=0)
            losses.append(loss_per_winner.mean())

        return torch.stack(losses).mean()

    return loss_fn