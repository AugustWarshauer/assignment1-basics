import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce, repeat
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        logits (torch.Tensor): shape (batch_size, sequence_length, vocab_size)
        targets (torch.LongTensor): A torch tensor.long of token IDs with shape (batch_size, sequence_length)
    Returns:
        average cross-entropy loss
    """
    logits -= torch.max(logits, -1, keepdim=True).values
    logit_sums = torch.log(reduce(torch.exp(logits), "... s v -> ... s", "sum"))
    target_logits = logits.gather(dim = -1, index=rearrange(targets, "... s -> ... s 1"))
    target_logits = rearrange(target_logits, "... s 1 -> ... s")
    return (logit_sums - target_logits).mean()


def learning_rate_schedule(t: int, max_lr: float, min_lr: float, warmup_iter: int, final_iter: int) -> float:
    """
    Cosine annealing learning rate scheduler
    
    Args:
        t (int): The current training iteration or step
        max_lr (float): The peak learning rate reached after warmup
        min_lr (float): The minimum learning rate reached by the end of scheduling
        warmup_iter (int): The number of warmup iterations during which the learning rate increases to max_lr
        final_iter (int): The iteration at which the schedule finishes and the learning rate reaches min_lr

    Returns:
        float: The learning rate value for iteration t
    """
    if t < warmup_iter:
        return max_lr*t/warmup_iter
    elif warmup_iter <= t <= final_iter:
        return min_lr + 0.5 * (1 + np.cos(np.pi * (t - warmup_iter)/(final_iter - warmup_iter))) * (max_lr - min_lr)
    else: #t > final_iter
        return min_lr


def gradient_clipping(params: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    params = [p for p in params if p.grad is not None]
    if not params:
        return
            
    l2norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in params])
    )

    if l2norm > max_l2_norm: 
        for p in params:
            p.grad *= max_l2_norm / (l2norm + eps)
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 1e-2, eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
            "t": 1,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            t = group.get("t", 1)  # Get iteration number from defaults
            beta1, beta2 = group["betas"]
            alpha_t = group["lr"] * ((1 - beta2**t) ** 0.5) / (1 - beta1**t)  # compute adjusted (unbiased) learning rate at iteration t
            group["t"] += 1
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data  # Get the gradient of loss with respect to p
                state = self.state[p]  # Get state associated with p
                p.data -= group["lr"] * group["weight_decay"] * p.data  # apply weight decay (regularization towards 0)
                m = beta1 * state.get("m", 0) + (1 - beta1) * grad  # First moment estimate update
                state["m"] = m
                v = beta2 * state.get("v", 0) + (1 - beta2) * grad**2  # Second moment estimate update
                state["v"] = v
                p.data -= alpha_t * m / (torch.sqrt(v) + group["eps"])  # Applying moment-adjusted weight updates
        return loss




"""training loop example"""
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD(weights, lr=1)
# for t in range(100):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.



