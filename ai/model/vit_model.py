import torch
import torch.nn as nn
from functools import partial
import math


#Patch Embedding Layer
class PatchEmbed(nn.Module):
    """Chia ảnh thành patch rồi embed thành vector."""
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 1, 128, 128]
        x = self.proj(x)               # -> [B, embed_dim, 8, 8]
        x = x.flatten(2)               # -> [B, embed_dim, 64]
        x = x.transpose(1, 2)          # -> [B, 64, embed_dim]
        return x


#FeedForward
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


#Self-Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.1, proj_drop=0.1):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: [B, heads, N, C/heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # attention score
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, heads, N, C/heads]
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#Transformer Block
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, dropout, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # residual 1
        x = x + self.mlp(self.norm2(x))   # residual 2
        return x


#Vision Transformer
class VisionTransformer(nn.Module):

    def __init__(self,img_size=128,patch_size=16,in_chans=1,num_classes=5,embed_dim=128,depth=6,num_heads=4,mlp_ratio=4.0,dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # vị trí cho từng patch
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # transformer layers
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes)

        self._reset_weights()

    def _reset_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # [B, 64, 128]

        # add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # lấy token đầu tiên để phân loại
        cls = x[:, 0]

        return self.head(cls)
