import torch
import torch.nn as nn
import math
from einops import rearrange, einsum, reduce, repeat
from collections.abc import Callable, Iterable
from typing import Optional


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


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, alpha: float, beta1: float = 0.9, beta2: float = 0.95, lambda_: float = 1e-2, eps: float = 1e-8):
        if alpha < 0:
            raise ValueError(f"Invalid learning rate: {alpha}")
        defaults = {
            "alpha": alpha,
            "beta1": beta1,
            "beta2": beta2,
            "lambda": lambda_,
            "eps": eps,
        }
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            t = group.get("t", 1) # Get iteration number from defaults
            alpha_t = group["alpha"]*((1-group["beta2"]**t)**0.5)/(1-group["beta1"]**t) #compute adjusted (unbiased) learning rate at iteration t
            group["t"] += 1
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data # Get the gradient of loss with respect to p
                state = self.state[p] # Get state associated with p
                p.data -= group["alpha"]*group["lambda"]*p.data #apply weight decay (regularization towards 0)
                m = group["beta1"]*state.get("m", 0)+(1-group["beta1"])*grad #First moment estimate update
                state["m"] = m
                v = group["beta2"]*state.get("v", 0)+(1-group["beta2"])*grad**2 #Second moment estimate update
                state["v"] = v
                p.data -= alpha_t * m / (math.sqrt(v) + group["eps"]) #Applying moment-adjusted weight updates
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



