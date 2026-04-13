import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce, repeat
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np
from tokenizer import Tokenizer
from transformer import softmax

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


def data_loading(x: np.typing.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (numpy array of ints): integer array with token IDs
        batch_size (int)
        context_length (int)
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0')
    Returns:
        a pair of tensors of shape (batch_size, context_length): the sampled input sequences and the corresponding next-token targets
    """
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs = np.stack([x[i:i + context_length] for i in start_indices])
    targets = np.stack([x[i + 1:i + context_length + 1] for i in start_indices])
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    """
    Args:
        model (torch.nn.Module): The model whose weights to serialize
        optimizer (torch.optim.Optimizer): The optimizer whose state to serialize
        iteration (int): The current training iteration number
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to write the checkpoint to
    """
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to load the checkpoint from
        model (torch.nn.Module): The model to restore weights into
        optimizer (torch.optim.Optimizer): The optimizer to restore state into
    Returns:
        int: The iteration number saved in the checkpoint
    """
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def decoding(model: nn.Module, tokenizer: Tokenizer, prompt: str, temp: float | None = None, top_p: float | None = None, max_generated: int | None = None, end_token: str = "<|endoftext|>") -> Iterable:
    """
    Args:
        model (nn.Module): our llm to give us logits 
        tokenizer (Tokenizer): our tokenizer
        prompt (str): provide model with sequence of prefix tokens 
        temp (float | None): temperature scaling of softmax (t->0 means softmax becomes one-hot vector)
        top_p (float | None): Must be between 0 and 1. top-p sampling where we modify sampling distribution by truncating lower prob tokens
        max_generated (int | None): max number of generated tokens
        end_token (str): token which we end text generation on
    Returns: 
        Iterable for next token vocab integer id
    """

    encoded_prompt = tokenizer.encode(prompt)
    
    if len(encoded_prompt) > model.d_model:
        encoded_prompt = encoded_prompt[-model.d_model:]
    elif len(encoded_prompt) < model.d_model:
        encoded_prompt = [tokenizer.reverse_vocab[b'\x00']] * (model.d_model - len(encoded_prompt)) + encoded_prompt

    x = torch.tensor(encoded_prompt).unsqueeze(0)
    t = 0 #start at 0 generations

    end_token_id = tokenizer.reverse_vocab[end_token.encode("utf-8")]  # adjust if your tokenizer needs bytes

    while max_generated is None or t < max_generated:
        logits = model(x) #returns size (batch_len, seq_len, vocab_size)
        logits = logits[0, -1] #pick out the end of seq_length output
        
        if temp is not None:
            if temp <= 0:
                raise ValueError("temp must be > 0")
            logits /= temp #applying temperature

        probs = softmax(logits, -1)

        if top_p is not None: #applying top_q
            if not (0 < top_p <= 1):
                raise ValueError("top_p must be in (0, 1]")
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            keep_mask = cumulative_probs <= top_p
            keep_mask[0] = True

            probs_top_p = torch.zeros_like(probs)
            probs_top_p[sorted_indices[keep_mask]] = probs[sorted_indices[keep_mask]]

            probs = probs_top_p

        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == end_token_id:
            break
        
        yield next_token
        t+=1


        x = torch.cat([x[:, 1:], next_token.view(x.size(0), 1)], dim=1)



"""training loop example"""
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD(weights, lr=1)
# for t in range(100):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.



