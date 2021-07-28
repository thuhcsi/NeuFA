import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .modules import Encoder, ReferenceEncoder, Decoder
from .attention import BidirectionalAttention, BidirectionalAdditiveAttention

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

class BidirectionalAttentionTest(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.encoder1 = Encoder(hparams.text_encoder)
        self.encoder2 = Encoder(hparams.text_encoder)
        self.attention = BidirectionalAdditiveAttention(hparams.text_encoder.output_dim, hparams.text_encoder.output_dim, hparams.attention.dim)
        #self.attention = BidirectionalAttention(hparams.text_encoder.output_dim, hparams.text_encoder.output_dim, hparams.attention.dim)
        self.decoder1 = Decoder(hparams.text_decoder)
        self.decoder2 = Decoder(hparams.text_decoder)
        self.softmax = nn.Softmax(-1)

        self.cross_entrophy = torch.nn.CrossEntropyLoss()

    def forward(self, texts):
        text_lengths = [i.shape[0] for i in texts]
        texts = pad_sequence(texts, batch_first=True)
        texts1 = self.encoder1(texts, text_lengths)
        texts2 = self.encoder2(texts, text_lengths)

        text1, texts2, w1, w2 = self.attention(texts1, texts2)

        texts1 = self.decoder1(texts1, text_lengths)
        texts2 = self.decoder2(texts2, text_lengths)

        texts1 = self.softmax(texts1)
        texts2 = self.softmax(texts2)

        return texts1, texts2, w1[0][:text_lengths[0], :text_lengths[0]], w2[0][:text_lengths[0], :text_lengths[0]]

    def loss(self, p_texts1, p_texts2, texts):
        text_lengths = [i.shape[0] for i in texts]

        p_texts1 = [p_text[:text_lengths[i]] for i, p_text in enumerate(p_texts1)]
        p_texts2 = [p_text[:text_lengths[i]] for i, p_text in enumerate(p_texts2)]

        p_texts1 = torch.cat(p_texts1)
        p_texts2 = torch.cat(p_texts2)

        texts = torch.cat(texts)

        return self.cross_entrophy(p_texts1, texts) + self.cross_entrophy(p_texts2, texts)

if __name__ == '__main__':
    import os
    from dataset import LibriSpeech
    from hdfs import HDFSLoader, HDFSCollate
    from hparams import test

    device = 'cuda:6'
    batch_size = 2

    dataset = LibriSpeech(os.path.expanduser('~/datasets/LibriSpeech/packed/LibriSpeech'))
    data_loader = HDFSLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=HDFSCollate(device), drop_last=True, num_readers=batch_size)

    for batch in data_loader:
        model = BidirectionalAttentionTest(test)
        model.to(device)
        output = model(batch[0])
        print([i.shape for i in output])
        loss = model.loss(*output[:2], batch[0])
        print(loss)
        break
