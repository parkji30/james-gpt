import torch
import torch.nn.functional as F
from torch import nn


def softmax(x):
    x = (
        x - x.max(dim=-1, keepdim=True).values
    )  # this makes largest value -> 0 and exp(0) = 1.
    exps = torch.exp(x)
    return exps / torch.sum(exps, dim=-1, keepdim=True)


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int, num_heads: int):

        super().__init__()

        assert embedding_dim % num_heads == 0  # ensure we can split and recombine.

        self.d_k = embedding_dim // num_heads

        self.q_embed = nn.Linear(in_dim, embedding_dim)
        self.k_embed = nn.Linear(in_dim, embedding_dim)
        self.v_embed = nn.Linear(in_dim, embedding_dim)

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        B, L, C = x.shape

        # shape is (B, L, H, d_k) after reshape
        # shape is (B, H, L, d_k) after reshape
        q = self.q_embed(x).reshape(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.k_embed(x).reshape(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v_embed(x).reshape(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # memories
        # fill fake scores first
        # qk_values = (q @ k.transpose(-2, -1)) / self.d_k**0.5
        #
        # Create lower triangular attention
        # mask = torch.tril(
        #     torch.ones(
        #         qk_values.size(-2),
        #         qk_values.size(-1),
        #         device=qk_values.device,
        #         dtype=torch.bool,
        #     )
        # )
        #
        # Fill the forbidden spots with -inf.
        # qk_values = qk_values.masked_fill(~mask, float("-inf"))
        #
        # shape is (B, H, L, d_k)
        # attention_scores = torch.vmap(softmax)(qk_values) @ v
        attention_scores = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        # Reshape this back into (B, L, C)
        attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(
            B, L, self.embedding_dim
        )

        # linear layer to fuse the learned heads.
        attention_scores = self.out_proj(attention_scores)

        return attention_scores


class MLP(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 2048):

        super().__init__()

        self.act = nn.GELU()
        self.linear_in = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)

        self.linear_out = nn.Linear(in_features=out_dim, out_features=in_dim, bias=True)

    def forward(self, x):

        x = self.linear_in(x)
        x = self.act(x)
        x = self.linear_out(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, num_heads: int):
        super().__init__()

        self.mlp = MLP(embed_dim, mlp_dim)

        self.causal_attention = MultiHeadCausalSelfAttention(
            embed_dim, embed_dim, num_heads
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # Causal Attention Block
        x = x + self.causal_attention((self.ln1(x)))

        # FF Block
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(
        self, vocab_size, num_heads, embed_dim, mlp_dim, context_length, decoder_blocks
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(context_length, embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim, mlp_dim=mlp_dim, num_heads=num_heads
                )
                for _ in range(decoder_blocks)
            ]
        )

        self.logits = nn.Linear(embed_dim, vocab_size)
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # we need to get the positions
        B, T = x.shape  # T for tokens
        position_indices = torch.arange(T, device=x.device)

        token_embedding = self.token_embedding(x)
        position_embedding = self.positional_embedding(position_indices)

        # Merge the token embeddings.
        x = token_embedding + position_embedding

        # pass through GPT layer
        for block in self.decoder_blocks:
            x = block(x)

        # Layernorm before logits
        x = self.ln_f(x)

        # Create logits
        x = self.logits(x)

        return x


if __name__ == "__main__":
    from config import GPT_CONFIG

    model = GPT(**GPT_CONFIG)
