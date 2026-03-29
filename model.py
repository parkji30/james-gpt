import torch
from torch import nn
import torch.nn.functional as F

def softmax(x):
    x = x - x.max() #this makes largest value -> 0 and exp(0) = 1.
    exps = torch.exp(x)
    return exps / torch.sum(exps)


class CausalSelfAttention(nn.Module):

    def __init__(self, in_dim, embedding_dim):
        
        super().__init__()

        self.q_embed = nn.Linear(in_dim, embedding_dim)
        self.k_embed = nn.Linear(in_dim, embedding_dim)
        self.v_embed = nn.Linear(in_dim, embedding_dim)

        self.dim_k = embedding_dim ** 0.5

    def forward(self, x):
        # Assume shape is (B, L, C)
        q = self.q_embed(x)
        k = self.k_embed(x)
        v = self.v_embed(x)
        
        # fill fake scores first
        scores = ((q @ k.transpose(-2, -1)) / self.dim_k)
        
        # Create lower triangular attention
        mask = torch.tril(
            torch.ones(scores.size(-2), scores.size(-1), device=scores.device, dtype=torch.bool)
        )

        # Fill the forbidden spots with -inf.
        scores = scores.masked_fill(~mask, float("-inf"))

        attention_scores = torch.vmap(softmax)(scores) @ v

        return attention_scores


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

    def __init__(self, embed_dim:int = 512, mlp_dim:int=2048):
        super().__init__()

        self.mlp = MLP(embed_dim, mlp_dim)

        self.causal_attention = CausalSelfAttention()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # Causal Attention Block
        x = x + self.causal_attention((self.ln1(x)))

        # FF Block
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):

    def __init__(self):

        self.positional_embedding = nn.Embedding

    def forward(self, x):

        ...


if __name__ == '__main__':
    model = GPT()
