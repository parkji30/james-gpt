import torch
from torch import nn
import torch.nn.functional as F

def softmax(x):
    x = x - x.max()
    exps = torch.exp(x)
    return exps / torch.sum(x)

class SelfAttention(nn.Module):

    def __init__(self, in_dim, embedding_dim):
        
        super().__init__()

        self.q_embed = nn.Linear(in_dim, embedding_dim)
        self.k_embed = nn.Linear(in_dim, embedding_dim)
        self.v_embed = nn.Linear(in_dim, embedding_dim)

        self.dim_k = embedding_dim ** 0.5

    def forward(self, x):

        q = self.q_embed(x)
        k = self.k_embed(x)
        v = self.v_embed(x)
        
        self_attention = F.softmax((q @ k.transpose(-2, -1)) / self.dim_k, dim=-1) @ v
        return self_attention


class MLP(nn.Module):

    def __init__(self, in_dim: int = 512, out_dim: int = 2048):

        super().__init__()

        self.act = nn.GELU()
        self.linear_in = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=True
        )

        self.linear_out = nn.Linear(
            in_features=out_dim,
            out_features=in_dim,
            bias=True
        )

    def forward(self, x):

        x = self.linear_in(x)
        x = self.act(x)
        x = self.linear_out(x)

        return x
            
class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        ...


class GPT(nn.Module):

    def __init__(self):

        self.positional_embedding = nn.Embedding

    def forward(self, x):

        ...


if __name__ == '__main__':
    model = GPT()
