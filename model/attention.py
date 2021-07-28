import torch
from torch import nn

class BidirectionalAttention(nn.Module):

    def __init__(self, k1_dim, k2_dim, v1_dim, v2_dim, attention_dim):
        super().__init__()
        self.k1_layer = nn.Linear(k1_dim, attention_dim)
        self.k2_layer = nn.Linear(k2_dim, attention_dim)
        self.score_layer = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, k1, k2, v1, v2):
        k1 = self.k1_layer(k1)
        k2 = self.k2_layer(k2)
        score = self.tanh(torch.bmm(k1, k2.transpose(1, 2)))
        w1 = self.softmax(score.transpose(1, 2)).squeeze(-1)
        w2 = self.softmax(score).squeeze(-1)
        o1 = torch.bmm(w1, v1)
        o2 = torch.bmm(w2, v2)
        return o1, o2, w1, w2

class BidirectionalAdditiveAttention(nn.Module):

    def __init__(self, k1_dim, k2_dim, v1_dim, v2_dim, attention_dim):
        super().__init__()
        self.k1_layer = nn.Linear(k1_dim, attention_dim)
        self.k2_layer = nn.Linear(k2_dim, attention_dim)
        self.score_layer = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, k1, k2, v1, v2):
        k1 = self.k1_layer(k1).repeat(k2.shape[1], 1, 1, 1).permute(1,2,0,3)
        k2 = self.k2_layer(k2).repeat(k1.shape[1], 1, 1, 1).permute(1,0,2,3)
        score = self.score_layer(self.tanh(k1 + k2))
        w1 = self.softmax(score.transpose(1, 2)).squeeze(-1)
        w2 = self.softmax(score).squeeze(-1)
        o1 = torch.bmm(w1, v1)
        o2 = torch.bmm(w2, v2)
        return o1, o2, w1, w2
