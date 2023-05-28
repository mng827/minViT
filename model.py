import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NewMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(NewMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(0.2)
        self.projection_dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.dim, dim=2)  # (B, T, C)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)

        attention = q @ k.transpose(-2, -1)  # (B, num_heads, T, head_dim) x (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        attention = F.softmax(attention * math.sqrt(self.head_dim), dim=-1)
        attention = self.attention_dropout(attention)
        x = attention @ v  # (B, num_heads, T, T) x (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)

        x = x.transpose(1, 2).reshape(B, T, C)  # Concatenate heads
        x = self.projection(x)
        x = self.projection_dropout(x)

        return x


class Block(nn.Module):
    """ A Transformer block """

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.model.embed_dim)
        self.attn = NewMultiHeadAttention(cfg.model.embed_dim, cfg.model.num_heads)
        self.ln_2 = nn.LayerNorm(cfg.model.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.model.embed_dim, 4 * cfg.model.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.model.embed_dim, cfg.model.embed_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PatchEmbed(nn.Module):
    """ Project an image into patch embeddings """
    def __init__(self, in_dim, embed_dim, patch_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # (B, embed_dim, num_patches_y, num_patches_x)
        x = x.reshape(B, self.embed_dim, -1).transpose(1, 2)

        return x


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(cfg.model.in_dim, cfg.model.embed_dim, cfg.model.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.model.num_patches, cfg.model.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.model.embed_dim))
        self.embed_dropout = nn.Dropout(0.2)

        self.transformer_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.model.num_layers)])
        self.ln = nn.LayerNorm(cfg.model.embed_dim)
        self.head = nn.Linear(cfg.model.embed_dim, cfg.model.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        B = x.shape[0]
        x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)
        x = self.embed_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln(x)
        # Just use the class token. Note that other models do not concatenate
        # a class token and just use the mean of all tokens (global pooling).
        x = self.head(x[:,0,:])

        return x
