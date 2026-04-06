import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce, repeat

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        y=Wx linear

        Params:
            in_features (int): final dimension of the input 
            out_features (int): final dimension of the output
            device (torch.device | None):  Device to store the parameters on
            dtype (torch.dtype | None):  Data type of the parameters
        """
        super().__init__()
        stdv = (2.0/(in_features+out_features))**0.5
        self.weights = nn.Parameter(nn.init.trunc_normal_(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            mean = 0.0,
            std = stdv,
            a = -3*stdv,
            b = 3*stdv
            )) # W = R^(d_out x d_in) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input
        """
        #use einops
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Params:
            num_embeddings (int):  Size of the vocabulary
            embedding_dim (int):  Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device (None):  Device to store the parameters on
            dtype: torch.dtype (None):  Data type of the parameters
        """
        super().__init__()
        self.embedding_matrix = nn.Parameter(nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
            mean = 0.0,
            std = 1,
            a = -3.0,
            b = 3.0
        )) #embedding matrix of shape (vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs

        Params:
            token_ids (torch.LongTensor): A torch tensor.long of token IDs with shape (batch_size, sequence_length)
        Returns: 
            shape (batch_size, sequence_length, d_model)
        """
        return self.embedding_matrix[token_ids]


class RMSnorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.rmsnorm_gains = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) #shape of (dmodel)
        self.eps = eps
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        params: 
            x (torch.Tensor): is of shape (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_squares = reduce(x * x, "... d_model -> ... 1", "mean")
        rms = (mean_squares + self.eps)**0.5
        result = x*self.rmsnorm_gains/rms

        return result.to(in_dtype)


class PositionwiseFeedForward(nn.Module):
    """ SwiGLU feed-forward network """
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        if d_ff == None:
            self.d_ff = 64*round(((8/3)*d_model)/64) # sets d_ff to 8/3*d_model to the nearest multiple of 64
        else: 
            self.d_ff = d_ff
        
        self.W1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype) #first linear layer
        self.W2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype) #second linear layer
        self.W3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype) #gate activations

    @staticmethod
    def _SiLu(x: torch.Tensor) -> torch.Tensor:
        return x*torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        params:
            x (torch.Tensor): input of shape (batch_size, sequence_length, d_model)
        """
        return self.W2(self._SiLu(self.W1(x))*self.W3(x))
    

class RotaryPositionalEmbedding(nn.Module):
    """
    applies RoPE to the input tensor. DISCLOSURE: I ended up getting stuck on matrix stuff and referenced an efficient implementation online to follow
    intuition: https://www.youtube.com/watch?v=qKUobBR5R1A
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None): 
        """
        Params:
            theta: float Θ value for the RoPE
            d_k: int  dimension of query and key vectors
            max_seq_len: int  Maximum sequence length that will be input
            device: torch.device | None = None  Device to store the buffer on
        """
        super().__init__()
        self.d_k = d_k
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        theta = 1.0 / (theta**((2*torch.arange(1, self.d_k // 2 + 1, device=device)-2)/self.d_k))
        thetas = einsum(positions, theta, "p, t -> p t")
        #thetas2 = torch.cat([thetas, thetas], dim=1) #This is the way it was done in stuff i looked at but fails cs336 test
        thetas2 = repeat(thetas, "p t -> p (t repeat)", repeat = 2)
        self.register_buffer('sin', torch.sin(thetas2), persistent=False)
        self.register_buffer('cos', torch.cos(thetas2), persistent=False)       

    # def _neg_half(self, x: torch.Tensor): #This is the way it was done in stuff i looked at but fails cs336 test
    #     return torch.cat([-x[..., self.d_k//2:], x[..., :self.d_k//2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]
    def _neg_half(self, x: torch.Tensor):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Processes an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape
        """
        
        return x*self.cos[token_positions]+self._neg_half(x)*self.sin[token_positions]
    

def softmax(x: torch.Tensor, i: int):
    """
    Params:
        x (torch.Tensor)
        i (int): dimension to apply softmax to
    """ 
    x -= torch.max(x, i, keepdim=True).values
    x = torch.exp(x)
    return x/torch.sum(x, i, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    """
    implements: Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    intuition: attention is all you need paper, zero to hero karpathy, https://medium.com/@saraswatp/understanding-scaled-dot-product-attention-in-transformer-models-5fe02b0f150c

    Params:
        K (torch.Tensor): shape (batch_size, ..., seq_len, d_k) 
        Q (torch.Tensor): shape (batch_size, ..., seq_len, d_k)
        V (torch.Tensor): shape (batch_size, ..., seq_len, d_v)
        mask (torch.Tensor | None): boolean mask of shape (seq_len, seq_len)
    Returns:
        output with shape (batch_size, ..., seq_len, d_v)
    """
    d_k = K.shape[-1] #key and query dimension
    cos_similarity = einsum(K, Q, "batch ... k_seq_len d_k, batch ... q_seq_len d_k -> batch ... q_seq_len k_seq_len")/d_k**0.5

    if mask is not None:
        cos_similarity.masked_fill_(mask.logical_not(), float("-inf"))

    return einsum(softmax(cos_similarity, -1), V, "batch ... q_seq_len k_seq_len, batch ... k_seq_len d_v -> batch ... q_seq_len d_v")


class MultiHead_Self_Attention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        d_model (int): Dimensionality of the Transformer block inputs
        num_heads (int): Number of heads to use in multi-head self-attention
        """

        super().__init__()

        self.d_k = int(d_model/num_heads) #d_k = d_v
        self.num_heads = num_heads

        self.W_q = Linear(num_heads*self.d_k, d_model, device=device, dtype=dtype)
        self.W_k = Linear(num_heads*self.d_k, d_model, device=device, dtype=dtype)
        self.W_v = Linear(num_heads*self.d_k, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, num_heads*self.d_k, device=device, dtype=dtype)

        self.RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        Params:
            x (torch.Tensor): has shape (batch_size, ..., seq_len, d_model)
            token_positions (torch.Tensor): has shape (..., sequence_length) and is a 1d Tensor of integers
        """
        seq_len = x.shape[-2]

        q = rearrange(self.W_q(x), "b ... seq_len (h d_k) -> b ... h seq_len d_k", h = self.num_heads)
        k = rearrange(self.W_k(x), "b ... seq_len (h d_k) -> b ... h seq_len d_k", h = self.num_heads)
        v = rearrange(self.W_v(x), "b ... seq_len (h d_k) -> b ... h seq_len d_k", h = self.num_heads)
        

        q = self.RoPE(q, token_positions)
        k = self.RoPE(k, token_positions)
                    
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
        attention = scaled_dot_product_attention(q, k, v, causal_mask) #(batch_size, ..., seq_len, d_v)
        attention = rearrange(attention, "batch_size ... h seq_len d_v -> batch_size ... seq_len (h d_v)")

        o = self.W_o(attention)
        return o



    





