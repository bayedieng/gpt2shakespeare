import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

N_HEADS = 12
D_MODEL = 768
D_FF = D_MODEL * 4
N_LAYERS = 4
P_DROPOUT = 0.1
MAX_SEQ_LEN = 2048
VOCAB_SIZE = tokenizer.get_vocab_size()

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, D_MODEL)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.pos_emb(positions)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_model = D_MODEL
        self.d_k = self.d_model // self.n_heads
        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.linear_w_out = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(P_DROPOUT)
        
    def split(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)
        mask = torch.tril(
            torch.ones((q.size(-2), q.size(-2)), device=q.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, v)

    def forward(self, x):
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        q, k, v = self.split(q), self.split(k), self.split(v)
        attn_out = self.attention(q, k, v)
        batch_size, _, seq_len, d_k = attn_out.size()
        d_model = self.n_heads * d_k
        multi_head = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.linear_w_out(multi_head)

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(D_MODEL, D_FF)
        self.l2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x):
        x = self.l1(x)
        x = F.gelu(x)
        x = self.l2(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = FFN()
        self.attention = Attention()
        self.layer_norm1 = nn.LayerNorm(D_MODEL)
        self.layer_norm2 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, x):
        attn_out = self.attention(self.layer_norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.layer_norm2(x))
        x = x + self.dropout(ffn_out)
        return x

class Gpt2Shakespeare(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embd = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_encoding = LearnedPositionalEmbedding()
        self.emb_dropout = nn.Dropout(P_DROPOUT)
        self.blocks = nn.ModuleList([Block() for i in range(N_LAYERS)])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.final_layer = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.text_embd(x) + self.pos_encoding(x)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block.forward(x)

        x = self.final_norm(x)
        x = self.final_layer(x)
        return x
