import torch
import torchsort



def spearmanr2(pred, target, **kw):
    "From https://github.com/teddykoker/torchsort/blob/main/README.md"
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()



def soft_rank_approx(x, tau=1.0):
    """
    A simple soft rank approximation using pairwise comparisons.
    Args:
        x: tensor of shape (..., n)
        tau: temperature (larger = softer, smaller = closer to true ranks)
    Returns:
        approx ranks same shape as x
    """
    diff = x.unsqueeze(-1) - x.unsqueeze(-2)
    # pairwise sigmoid scores
    P = torch.sigmoid(diff / tau)
    # sum of how many values each element is less than
    r = P.sum(dim=-1) + 0.5  # +0.5 to approximate average rank
    return r

def spearman_soft(x, y, tau=1.0):
    rx = soft_rank_approx(x, tau)
    ry = soft_rank_approx(y, tau)

    # center
    rxc = rx - rx.mean(-1, keepdim=True)
    ryc = ry - ry.mean(-1, keepdim=True)

    # normalize
    rxn = rxc / (rxc.norm(dim=-1, keepdim=True) + 1e-8)
    ryn = ryc / (ryc.norm(dim=-1, keepdim=True) + 1e-8)

    return (rxn * ryn).sum(dim=-1)




def spearman_corr_differentiable(pred: torch.Tensor, target: torch.Tensor,
                                 regularization_strength: float = 1.0,
                                 regularization: str = "l2"):
    """
    REQUIRES TORCHSORT
    Compute a differentiable Spearman correlation coefficient between pred and target.
    Works on [batch_size, n] tensors; preserves gradients for backprop.
    """
    # Soft ranks
    pred_rank = torchsort.soft_rank(pred, regularization="l2", regularization_strength=regularization_strength)
    target_rank = torchsort.soft_rank(target, regularization="l2", regularization_strength=regularization_strength)

    # Center and normalize
    pred_rank = pred_rank - pred_rank.mean(dim=-1, keepdim=True)
    pred_rank = pred_rank / (pred_rank.norm(dim=-1, keepdim=True) + 1e-8)
    target_rank = target_rank - target_rank.mean(dim=-1, keepdim=True)
    target_rank = target_rank / (target_rank.norm(dim=-1, keepdim=True) + 1e-8)

    # Spearman = dot product of normalized ranks
    return (pred_rank * target_rank).sum(dim=-1)
