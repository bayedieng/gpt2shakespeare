import torch
import torch.nn as nn
import torch.nn.functional as F
import math

N_HEADS = 12
D_MODEL = 768

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_model = D_MODEL
        self.d_k = self.d_model // self.n_heads
        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.linear_final_layer = nn.Linear(self.d_model, self.d_model)
        self.linear_w_out = nn.Linear(self.d_model, self.d_model)

    def split(self, x):
        '''
        input -> [batch_size, seq_len, d_model]
        return -> [batch_size, head, seq_len, d_k]
        '''
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, self.n_heads, seq_len, self.d_k)
        return x

    def attention(self, q, k, v):
        k = k.transpose(2, 3)
        qk_mul = torch.matmul(q, k)
        attn_out = F.softmax(torch.matmul(qk_mul / math.sqrt(self.d_k), v), dim=1)
        return attn_out


    def forward(self, x):
        # [batch_size, seq_len, d_model]
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        # split into heads
        q,k,v = self.split(q), self.split(k), self.split(v)

        # apply attention and concat output
        attn_out = self.attention(q, k, v)

        # concat out as multi-head
        batch_size, n_heads, seq_len, d_k = attn_out.size()
        d_model = n_heads * d_k
        multi_head = attn_out.view(batch_size,seq_len, d_model)

        # project into the two final layers
        multi_head = self.linear_w_out(multi_head)
        out = self.linear_final_layer(multi_head)
        return out
