import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .modules import TacotronEncoder, Tacotron2Encoder, ReferenceEncoder, Decoder, GumbelSoftmax, MixedGumbelSoftmax
from .attention import BidirectionalAttention, BidirectionalAdditiveAttention
from .position import PositionalEncoding

class BidirectionalAttention(BidirectionalAttention):

    def __init__(self, input1_dim, input2_dim, attention_dim):
        super().__init__(input1_dim, input2_dim, input1_dim, input2_dim, attention_dim)
        self.softmax1 = GumbelSoftmax(dim=1)
        self.softmax2 = GumbelSoftmax(dim=1)

class BidirectionalAdditiveAttention(BidirectionalAdditiveAttention):

    def __init__(self, input1_dim, input2_dim, attention_dim):
        super().__init__(input1_dim, input2_dim, input1_dim, input2_dim, attention_dim)

class Encoder(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.embedding = torch.nn.Embedding(hparams.num_symbols, hparams.embedding_dim)

    def forward(self, inputs, input_lengths=None):
        return self.embedding(inputs)

class Decoder(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.linear = torch.nn.Linear(hparams.input_dim, hparams.output_dim)

    def forward(self, inputs, input_lengths):
        return self.linear(inputs)

class BidirectionalAttentionTest(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.encoder1 = Tacotron2Encoder(hparams.text_encoder)
        self.encoder2 = Tacotron2Encoder(hparams.text_encoder)
        self.attention = BidirectionalAdditiveAttention(hparams.text_encoder.output_dim, hparams.text_encoder.output_dim, hparams.attention.dim)
        self.decoder1 = Decoder(hparams.text_decoder)
        self.decoder2 = Decoder(hparams.text_decoder)
        self.positional_encoding = PositionalEncoding(hparams.text_encoder.output_dim)

        self.cross_entrophy = torch.nn.CrossEntropyLoss()

    def forward(self, texts):
        text_lengths = [i.shape[0] for i in texts]
        texts = pad_sequence(texts, batch_first=True)
        texts1 = self.encoder1(texts, text_lengths)
        texts1_pe = self.positional_encoding(texts1)
        texts2 = self.encoder2(texts, text_lengths)
        texts2_pe = self.positional_encoding(texts2)

        text1, texts2, w1, w2 = self.attention(texts1_pe, texts2_pe, texts1, texts2, text_lengths, text_lengths)

        texts1 = self.decoder1(texts1, text_lengths)
        texts2 = self.decoder2(texts2, text_lengths)

        #return texts1, texts2, w1[0][:text_lengths[0], :text_lengths[0]], w2[0][:text_lengths[0], :text_lengths[0]]
        return texts1, texts2, w1, w2

    def diagonal_attention_loss(self, w1, w2, k1_lengths, k2_lengths):
        mask = torch.zeros(attention.shape, dtype=torch.int).detach().to(score.device)

        for i, l1, l2 in enumerate(zip(k1_lengths, k2_lengths)):

            mask[i,l:,:] += 1
        for i, l in enumerate(k2_lengths):
            mask[i,:,l:] += 1


    def loss(self, p_texts1, p_texts2, texts):
        text_lengths = [i.shape[0] for i in texts]

        p_texts1 = [p_text[:text_lengths[i]] for i, p_text in enumerate(p_texts1)]
        p_texts2 = [p_text[:text_lengths[i]] for i, p_text in enumerate(p_texts2)]

        p_texts1 = torch.cat(p_texts1)
        p_texts2 = torch.cat(p_texts2)

        texts = torch.cat(texts)

        return self.cross_entrophy(p_texts1, texts) + self.cross_entrophy(p_texts2, texts)

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

class BidirectionalAttentionTest2(BidirectionalAttentionTest):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.attention = BidirectionalAttention(hparams.text_encoder.output_dim, hparams.text_encoder.output_dim, hparams.attention.dim)

if __name__ == '__main__':
    import os
    from dataset import LibriSpeechText
    from hdfs import HDFSLoader, HDFSCollate
    from hparams import test

    device = 'cuda:1'
    batch_size = 2

    dataset = LibriSpeechText('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    data_loader = HDFSLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=HDFSCollate(device), drop_last=True, num_readers=batch_size)

    for batch in data_loader:
        model = BidirectionalAttentionTest2(test)
        model.to(device)
        output = model(batch[0])
        #print([i.shape for i in output])
        print(model.loss(*output[:2], batch[0]))
        print(model.attention_loss(*output[2:4]))
        break
