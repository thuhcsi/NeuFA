import torch
from torch import nn

class GumbelSoftmax(nn.Module):

    def __init__(self, hard=True, **kwargs):
        super().__init__()
        self.hard = hard
        self.kwargs = kwargs

    def forward(self, inputs):
        return nn.functional.gumbel_softmax(inputs, hard=self.hard, **self.kwargs)

class MixedGumbelSoftmax(nn.Module):

    def __init__(self, hard_rate=0.5, **kwargs):
        super().__init__()
        if 'hard' in kwargs:
            del kwargs['hard']
        self.kwargs = kwargs
        self.hard_rate = hard_rate

    def forward(self, inputs):
        if self.hard_rate == 0:
            return nn.functional.gumbel_softmax(inputs, hard=False, **self.kwargs)
        if self.hard_rate == 1:
            return nn.functional.gumbel_softmax(inputs, hard=True, **self.kwargs)
        soft = nn.functional.gumbel_softmax(inputs, hard=False, **self.kwargs)
        hard = nn.functional.gumbel_softmax(inputs, hard=True, **self.kwargs)
        random = torch.rand(soft.shape[:-1], device=soft.device)
        random[random <= self.hard_rate] = 0
        random[random > self.hard_rate] = 1
        random = random.repeat(soft.shape[-1], 1, 1).permute(1, 2, 0)
        return hard * random + soft * (1 - random)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Prenet(nn.Module):

    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, activation=None):
        super().__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)

class BatchNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super().__init__()
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList([BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1, padding=k // 2, activation=self.relu) for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True, enforce_sorted=False)

        # (B, T_in, in_dim*2)
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs

class TacotronEncoder(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.embedding = nn.Embedding(hparams.num_symbols, hparams.embedding_dim)
        self.prenet = Prenet(hparams.embedding_dim, sizes=hparams.prenet.sizes)
        self.cbhg = CBHG(hparams.cbhg.dim, K=hparams.cbhg.K, projections=hparams.cbhg.projections)

    def forward(self, inputs, input_lengths=None):
        x = self.embedding(inputs)
        x = self.prenet(x)
        x = self.cbhg(x, input_lengths)
        return x

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Tacotron2Encoder(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.embedding = nn.Embedding(hparams.num_symbols, hparams.embedding_dim)
        convolutions = []
        for _ in range(hparams.cnn.num_layers):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.embedding_dim,
                         hparams.embedding_dim,
                         kernel_size=hparams.cnn.kernel_size, stride=1,
                         padding=int((hparams.cnn.kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.embedding_dim, int(hparams.embedding_dim / 2), 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x).transpose(1, 2)

        for conv in self.convolutions:
            x = torch.nn.functional.dropout(torch.nn.functional.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

class ContentEncoder(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        filters = [hparams.input_dim] + hparams.cnn.filters
        convs = (BatchNormConv1d(filters[i], filters[i + 1], hparams.cnn.kernel_size, 1, hparams.cnn.kernel_size//2) for i in range(len(hparams.cnn.filters)))
        self.convs = nn.Sequential(*convs)

        self.gru = nn.GRU(input_size=hparams.cnn.filters[-1], hidden_size=hparams.gru_dim, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(input_size=hparams.gru_dim * 2, hidden_size=hparams.gru_dim, bidirectional=True, batch_first=True)

    def forward(self, inputs, input_lengths):
        x = self.convs(inputs.transpose(1, 2)).transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x, _ = self.gru2(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x

class Decoder(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.lstm = nn.LSTM(input_size=hparams.input_dim, hidden_size=hparams.lstm_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hparams.lstm_dim * 2, hidden_size=hparams.lstm_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hparams.lstm_dim * 2, hparams.output_dim)

    def forward(self, inputs, input_lengths):
        x = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.linear(x)
        x = [x[i][:l] for i, l in enumerate(input_lengths)]
        return x

class Aligner(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.max_frames = hparams.max_frames
        #self.lstm = nn.GRU(input_size=6, hidden_size=hparams.lstm_dim, bidirectional=True, batch_first=True)
        #self.lstm2 = nn.GRU(input_size=hparams.lstm_dim * 2, hidden_size=hparams.lstm_dim, bidirectional=True, batch_first=True)

        #self.location_layer = LocationLayer(hparams.location_layer.attention_n_filters, hparams.location_layer.attention_kernel_size, hparams.location_layer.output_dim)
        filters = [hparams.input_dim] + hparams.cnn.filters
        #convs = (BatchNormConv1d(filters[i], filters[i + 1], hparams.cnn.kernel_size, 1, hparams.cnn.kernel_size//2) for i in range(len(hparams.cnn.filters)))
        #self.convs = nn.Sequential(*convs)
        convs = (BatchNormConv2d(filters[i], filters[i + 1], hparams.cnn.kernel_size, 1, tuple([i//2 for i in hparams.cnn.kernel_size])) for i in range(len(hparams.cnn.filters)))
        self.convs = nn.Sequential(*convs)

        self.linear = nn.Linear(filters[-1], 2)
        self.softmax = nn.Softmax(-1)

    def stack_attention(self, w1, w2):
        w1 = [i.T for i in w1]
        max_frames = max([i.shape[1] for i in w1])
        w1 = [torch.cat([i, torch.zeros(i.shape[0], max_frames - i.shape[1], device=i.device)], dim=-1) for i in w1]
        w2 = [torch.cat([i, torch.zeros(i.shape[0], max_frames - i.shape[1], device=i.device)], dim=-1) for i in w2]
        w1 = nn.utils.rnn.pad_sequence(w1, batch_first=True)
        w2 = nn.utils.rnn.pad_sequence(w2, batch_first=True)
        accumulated_w1 = torch.cumsum(w1, -1)
        accumulated_w2 = torch.cumsum(w2, -1)
        accumulated_w1_backward = torch.cumsum(w1.flip(-1), -1).flip(-1)
        accumulated_w2_backward = torch.cumsum(w2.flip(-1), -1).flip(-1)
        #x = torch.stack([w1, w2, accumulated_w1, accumulated_w2, accumulated_w1_backward, accumulated_w2_backward], dim=-1).permute(1, 0, 3, 2)
        #return torch.stack([self.convs(i) for i in x]).permute(1, 0, 3, 2)
        x = torch.stack([w1, w2, accumulated_w1, accumulated_w2, accumulated_w1_backward, accumulated_w2_backward], dim=-1).permute(0, 3, 1, 2)
        return self.convs(x).permute(0, 2, 3, 1)

    def forward(self, texts, w1, w2, text_lengths, mfcc_lengths):
        x = self.stack_attention(w1, w2)
        x = torch.sigmoid(self.linear(x).transpose(-1, -2))
        x = torch.cumsum(x, dim=-1)
        #x = torch.stack([torch.cumsum(x[:,:,0,:], dim=-1), torch.cumsum(x[:,:,1,:].flip(-1), dim=-1).flip(-1)], dim=-2)
        x = torch.tanh(x)
        x = [b[:l1, :, :l2] for b, l1, l2 in zip(x, text_lengths, mfcc_lengths)]
        return x

class Predictor(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.max_frames = hparams.max_frames
        self.lstm = nn.GRU(input_size=hparams.input_dim, hidden_size=hparams.lstm_dim, bidirectional=True, batch_first=True, dropout=0.5)
        self.lstm2 = nn.GRU(input_size=hparams.lstm_dim * 2, hidden_size=hparams.lstm_dim, bidirectional=True, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(hparams.lstm_dim * 2, 2)
        #self.location_layer = LocationLayer(hparams.location_layer.attention_n_filters, hparams.location_layer.attention_kernel_size, hparams.location_layer.output_dim)

    def clip_score(self, score, text_lengths, mfcc_lengths):
        middles = [torch.linspace(self.max_frames, self.max_frames + mfcc_lengths[i] - 1, text_length, dtype=torch.int) for i, text_length in enumerate(text_lengths)]
        tops = [i + self.max_frames for i in middles]
        bottoms = [i - self.max_frames for i in middles]

        clipped_score = []
        score = [torch.cat([torch.zeros(i.shape[0], self.max_frames, device=i.device), i, torch.zeros(i.shape[0], self.max_frames, device=i.device)], dim=-1) for i in score]
        clipped_score = []
        for i, top, bottom in zip(score, tops, bottoms):
            clipped_score.append([torch.cat([j[b:t], t.to(j.device).unsqueeze(-1), b.to(j.device).unsqueeze(-1)]) for j, t, b in zip(i, top, bottom)])
            clipped_score[-1] = torch.stack(clipped_score[-1], axis=0)
        return clipped_score

    def forward(self, texts, w1, w2, text_lengths, mfcc_lengths):
        w1 = [i.T for i in w1]
        clipped_w1 = self.clip_score(w1, text_lengths, mfcc_lengths)
        clipped_w2 = self.clip_score(w2, text_lengths, mfcc_lengths)
        clipped_score = [torch.cat([i, j], dim=-1) for i, j in zip(clipped_w1, clipped_w2)]
        #clipped_score = [torch.cat([i, j], dim=-1) for i, j in zip(clipped_w1, clipped_w1)]

        clipped_score = nn.utils.rnn.pad_sequence(clipped_score, batch_first=True)
        x = torch.cat([texts, clipped_score], dim=-1)
        x = nn.utils.rnn.pack_padded_sequence(x, text_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        boundries = torch.relu(self.linear(x))
        #boundries = torch.cumsum(boundries, dim=-1)
        boundries = [boundries[i, :l] for i, l in enumerate(text_lengths)]
        return boundries
