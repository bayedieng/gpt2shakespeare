import torch
import torch.nn as nn
import torch.nn.functional as F
import math

N_HEADS = 12
D_MODEL = 768
D_FF = D_MODEL * 4
N_LAYERS = 4


class PosEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        position = torch.arange(5000)  # 5000 from the torch docs
        div_term = torch.exp(torch.arange(0, D_MODEL))
        pe = torch.zeros(5000, 1, D_MODEL)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)]
        return x

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
          return torch.matmul(weights, v)

      def forward(self, x):
          q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
          q, k, v = self.split(q), self.split(k), self.split(v)
          attn_out = self.attention(q, k, v)
          batch_size, _, seq_len, d_k = attn_out.size()
          d_model = self.n_heads * d_k
          multi_head = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len,
  d_model)
          return self.linear_w_out(multi_head)

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(D_MODEL, D_FF)
        self.l2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

