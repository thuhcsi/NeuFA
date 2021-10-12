import torch
import math

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('div_term', div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, custom_position=None):
        if not custom_position is None:
            custom_position = custom_position.repeat(self.div_term.shape[0], 1, 1).permute(1, 2, 0)
            pe = torch.zeros(custom_position.shape[:-1] + (self.d_model, ))
            pe[:, :, 0::2] = torch.sin(custom_position * self.div_term)
            pe[:, :, 1::2] = torch.cos(custom_position * self.div_term)
            x = x + pe.to(x.device)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
