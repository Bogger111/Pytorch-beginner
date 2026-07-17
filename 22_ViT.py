import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchPositionEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channel,ebd_dim):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embedding = nn.Conv2d(
            in_channel,out_channels=ebd_dim,
            kernel_size=patch_size,stride=patch_size
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches+1, ebd_dim)
        )

        self.CLS_token = nn.Parameter(
            torch.randn(1, 1, ebd_dim)
        )


    def forward(self, x):
        B, C, H, W  =x.shape

        x = self.patch_embedding(x)

        x = x.flatten(2)

        x = x.transpose(1,2)

        CLS_token = self.CLS_token.expand(
            x.shape[0],
            -1,
            -1
        )

        x = torch.cat(
            [CLS_token,x],
            dim=1
        )

        x = x + self.position_embedding

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.W_o(out)

        return out, attn_weights

class VisionTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        attn_out, attn_weights = self.attention(
            self.norm1(x)
        )

        x = x + attn_out

        ffn_out =  self.ffn(
            self.norm2(x)
        )

        x = x + ffn_out

        return x, attn_weights

class MiniTransformerClassifier(nn.Module):
    def __init__(
            self, 
            img_size, 
            patch_size,
            in_channel,
            ebd_dim,  
            num_heads, 
            d_ff, 
            num_layers,
            num_classes,
        ):
        super().__init__()

        self.embedding = PatchPositionEmbedding(
            img_size, patch_size, 
            in_channel, ebd_dim
        )

        self.block = nn.ModuleList([
            VisionTransformerBlock(ebd_dim, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(ebd_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        all_weights = []

        for block in self.block:
            x, attn_weights = block(x)
            all_weights.append(attn_weights)

        cls_out = x[:, 0, :]

        logits = self.classifier(cls_out)

        return logits, all_weights
    


model = MiniTransformerClassifier(
    img_size=224,
    patch_size=16,
    in_channel=3,
    ebd_dim=768,
    num_heads=12,
    d_ff=2048,
    num_layers=6,
    num_classes=10
)


x=torch.randn(2,3,224,224)

y, weights=model(x)

print(y.shape)
print(len(weights))
print(weights[0].shape)

