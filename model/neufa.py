import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .modules import Tacotron2Encoder, ReferenceEncoder, ContentEncoder, Decoder, GumbelSoftmax
from .test import Encoder as TestEncoder
from .test import Decoder as TestDecoder
from .attention import BidirectionalAttention, BidirectionalAdditiveAttention
from .position import PositionalEncoding

class BidirectionalAttention(BidirectionalAttention):

    def __init__(self, input1_dim, input2_dim, attention_dim):
        super().__init__(input1_dim, input2_dim, input1_dim, input2_dim, attention_dim)
        #self.softmax2 = GumbelSoftmax(dim=1)

class BidirectionalAdditiveAttention(BidirectionalAdditiveAttention):

    def __init__(self, input1_dim, input2_dim, attention_dim):
        super().__init__(input1_dim, input2_dim, input1_dim, input2_dim, attention_dim)

class NeuFA_base(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.scale_factor = 1 / hparams.scale_factor
        self.layer_normalization = nn.LayerNorm(hparams.input.mfcc_dim)

        self.text_encoder = Tacotron2Encoder(hparams.text_encoder)
        self.speech_encoder = ContentEncoder(hparams.speech_encoder)
        #self.attention = BidirectionalAdditiveAttention(hparams.text_encoder.output_dim, hparams.speech_encoder.output_dim, hparams.attention.dim)
        self.attention = BidirectionalAttention(hparams.text_encoder.output_dim, hparams.speech_encoder.output_dim, hparams.attention.dim)
        self.text_decoder = Decoder(hparams.text_decoder)
        self.speech_decoder = Decoder(hparams.speech_decoder)
        self.positional_encoding_text = PositionalEncoding(hparams.text_encoder.output_dim)
        self.positional_encoding_speech = PositionalEncoding(hparams.speech_encoder.output_dim)

        self.mse = nn.MSELoss()
        self.cross_entrophy = torch.nn.CrossEntropyLoss()

    def forward(self, texts, mfccs):
        text_lengths = [i.shape[0] for i in texts]
        texts = pad_sequence(texts, batch_first=True)
        texts = self.text_encoder(texts, text_lengths)
        texts = torch.cat([texts, torch.zeros((texts.shape[0], 1, texts.shape[2])).to(texts.device)], axis=-2)

        mfcc_lengths = [i.shape[0] for i in mfccs]
        mfccs = pad_sequence(mfccs, batch_first=True)

        if self.scale_factor < 1:
            mfccs = nn.functional.interpolate(mfccs.transpose(1, 2), scale_factor=self.scale_factor).transpose(1, 2)
            mfcc_lengths = [int(i * self.scale_factor) for i in mfcc_lengths]

        normalized_mfccs = self.layer_normalization(mfccs)
        mfccs = self.speech_encoder(normalized_mfccs, mfcc_lengths)
        #mfccs = self.speech_encoder(mfccs, mfcc_lengths)
        mfccs = torch.cat([mfccs, torch.zeros((mfccs.shape[0], 1, mfccs.shape[2])).to(mfccs.device)], axis=-2)

        normalized_mfccs = [mfcc[:mfcc_lengths[i]] for i, mfcc in enumerate(normalized_mfccs)]

        texts_pe = self.positional_encoding_text(texts)
        mfccs_pe = self.positional_encoding_speech(mfccs)

        texts_at_frame, mfccs_at_text, w1, w2 = self.attention(texts_pe, mfccs_pe, texts, mfccs, text_lengths, mfcc_lengths)

        texts = self.text_decoder(mfccs_at_text, text_lengths)
        mfccs = self.speech_decoder(texts_at_frame, mfcc_lengths)

        return texts, mfccs, w1, w2, normalized_mfccs

    def text_loss(self, p_texts, texts):
        text_lengths = [i.shape[0] for i in texts]
        p_texts = [p_text[:text_lengths[i]] for i, p_text in enumerate(p_texts)]
        p_texts = torch.cat(p_texts)
        texts = torch.cat(texts)
        return self.cross_entrophy(p_texts, texts)

    def mfcc_loss(self, p_mfccs, mfccs):
        mfcc_lengths = [i.shape[0] for i in mfccs]
        p_mfccs = [p_mfcc[:mfcc_lengths[i]] for i, p_mfcc in enumerate(p_mfccs)]
        p_mfccs = torch.cat(p_mfccs)
        mfccs = torch.cat(mfccs)
        return self.mse(p_mfccs, mfccs)

    def attention_loss(self, w1, w2, alpha=1):
        loss = []
        for _w1, _w2 in zip(w1, w2):
            w = torch.maximum(_w1.T, _w2)
            a = torch.linspace(1e-6, 1, w.shape[0]).to(w.device).repeat(w.shape[1], 1).T
            b = torch.linspace(1e-6, 1, w.shape[1]).to(w.device).repeat(w.shape[0], 1)
            r1 = torch.maximum((a / b), (b / a))
            r2 = torch.maximum(a.flip(1) / b.flip(0), b.flip(0)/ a.flip(1))
            r = torch.maximum(r1, r2) - 1
            r = torch.tanh(alpha * r)
            loss.append(torch.mean(w * r.detach()))
        loss = torch.stack(loss)
        return torch.mean(loss)

if __name__ == '__main__':
    import os
    from dataset import LibriSpeech
    from hdfs import HDFSLoader, HDFSCollate
    from hparams import base

    device = 'cuda:1'
    batch_size = 4

    dataset = LibriSpeech('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    data_loader = HDFSLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=HDFSCollate(device), drop_last=True, num_readers=batch_size)

    for batch in data_loader:
        model = NeuFA_base(base)
        model.to(device)
        output = model(*batch)
        #print([i.shape for i in output])
        print(model.text_loss(output[0], batch[0]))
        print(model.mfcc_loss(output[1], output[-1]))
        print(model.attention_loss(output[2], output[3]))
        break
