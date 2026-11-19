import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim)
        q=q.transpose(2,1)

        k = k.reshape(B, T, self.num_kv_heads, self.head_dim)
        k=k.transpose(2,1)

        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)
        v=v.transpose(2,1)

        repeat_factor=self.num_heads//self.num_kv_heads
        k=torch.repeat_interleave(k,repeats=repeat_factor,dim=1)
        v=torch.repeat_interleave(v,repeats=repeat_factor,dim=1)
        d_k=self.head_dim
        scores=(q@k.transpose(-2,-1))/d_k**0.5
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores=scores.masked_fill(mask==0,float('-inf'))
        attention=torch.softmax(scores,dim=-1)
        output=attention@v
        output=output.transpose(1,2)
        output=output.reshape(B,T,self.num_heads*self.head_dim)
        output = self.output_proj(output)
        return torch.round(output,decimals=4)
