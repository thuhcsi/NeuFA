import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .modules import Encoder, ReferenceEncoder, Decoder
from .attention import BidirectionalAttention, BidirectionalAdditiveAttention
from .position import PositionalEncoding

class BidirectionalAttention(BidirectionalAttention):

    def __init__(self, input1_dim, input2_dim, attention_dim):
        super().__init__(input1_dim, input2_dim, input1_dim, input2_dim, attention_dim)

    def forward(self, input1, input2):
        return super().forward(input1, input2, input1, input2)

class BidirectionalAdditiveAttention(BidirectionalAdditiveAttention):

    def __init__(self, input1_dim, input2_dim, attention_dim):
        super().__init__(input1_dim, input2_dim, input1_dim, input2_dim, attention_dim)

    def forward(self, input1, input2):
        return super().forward(input1, input2, input1, input2)

class NeuFA_base(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.layer_normalization = nn.LayerNorm(hparams.input.mfcc_dim)

        self.text_encoder = Encoder(hparams.text_encoder)
        self.speech_encoder = ReferenceEncoder(hparams.speech_encoder)
        #self.attention = BidirectionalAdditiveAttention(hparams.text_encoder.output_dim, hparams.speech_encoder.output_dim, hparams.attention.dim)
        self.attention = BidirectionalAttention(hparams.text_encoder.output_dim, hparams.speech_encoder.output_dim, hparams.attention.dim)
        self.text_decoder = Decoder(hparams.text_decoder)
        self.speech_decoder = Decoder(hparams.speech_decoder)
        self.positional_encoding_text = PositionalEncoding(hparams.text_encoder.output_dim)
        self.positional_encoding_speech = PositionalEncoding(hparams.speech_encoder.output_dim)
        self.softmax = nn.Softmax(-1)

        self.mse = nn.MSELoss()
        self.cross_entrophy = torch.nn.CrossEntropyLoss()

    def forward(self, texts, mfccs):
        text_lengths = [i.shape[0] for i in texts]
        texts = pad_sequence(texts, batch_first=True)
        texts = self.text_encoder(texts, text_lengths)

        mfcc_lengths = [i.shape[0] for i in mfccs]
        mfccs = pad_sequence(mfccs, batch_first=True)
        normalized_mfccs = self.layer_normalization(mfccs)
        mfccs = self.speech_encoder(normalized_mfccs)

        texts = self.positional_encoding_text(texts)
        mfccs = self.positional_encoding_speech(mfccs)

        texts_at_frame, mfccs_at_text, w1, w2 = self.attention(texts, mfccs)

        texts = self.text_decoder(mfccs_at_text, text_lengths)
        mfccs = self.speech_decoder(texts_at_frame, mfcc_lengths)

        texts = self.softmax(texts)

        return texts, mfccs, w1[0][:mfcc_lengths[0], :text_lengths[0]], w2[0][:text_lengths[0], :mfcc_lengths[0]], normalized_mfccs

    def loss(self, p_texts, p_mfccs, texts, mfccs):
        text_lengths = [i.shape[0] for i in texts]
        mfcc_lengths = [i.shape[0] for i in mfccs]

        p_texts = [p_text[:text_lengths[i]] for i, p_text in enumerate(p_texts)]
        p_mfccs = [p_mfcc[:mfcc_lengths[i]] for i, p_mfcc in enumerate(p_mfccs)]
        mfccs = [p_mfcc[:mfcc_lengths[i]] for i, p_mfcc in enumerate(mfccs)]

        p_texts = torch.cat(p_texts)
        p_mfccs = torch.cat(p_mfccs)

        texts = torch.cat(texts)
        mfccs = torch.cat(mfccs)

        return self.mse(p_mfccs, mfccs) + self.cross_entrophy(p_texts, texts)

if __name__ == '__main__':
    import os
    from dataset import LibriSpeech
    from hdfs import HDFSLoader, HDFSCollate
    from hparams import base

    device = 'cuda:6'
    batch_size = 2

    dataset = LibriSpeech(os.path.expanduser('~/datasets/LibriSpeech/packed/LibriSpeech'))
    data_loader = HDFSLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=HDFSCollate(device), drop_last=True, num_readers=batch_size)

    for batch in data_loader:
        model = NeuFA_base(base)
        model.to(device)
        output = model(*batch)
        print([i.shape for i in output])
        loss = model.loss(*output[:2], batch[0], output[-1])
        print(loss)
        break
