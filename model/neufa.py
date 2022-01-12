import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .modules import Tacotron2Encoder, ContentEncoder, Decoder, Aligner, Predictor
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

        self.text_encoder = Tacotron2Encoder(hparams.text_encoder)
        self.speech_encoder = ContentEncoder(hparams.speech_encoder)
        #self.attention = BidirectionalAdditiveAttention(hparams.text_encoder.output_dim, hparams.speech_encoder.output_dim, hparams.attention.dim)
        self.attention = BidirectionalAttention(hparams.attention.text_input_dim, hparams.attention.speech_input_dim, hparams.attention.dim)
        self.text_decoder = Decoder(hparams.text_decoder)
        self.speech_decoder = Decoder(hparams.speech_decoder)
        self.positional_encoding_text = PositionalEncoding(hparams.text_encoder.output_dim)
        self.positional_encoding_speech = PositionalEncoding(hparams.speech_encoder.output_dim)
        self.aligner = Aligner(hparams.aligner)
        #self.aligner = Predictor(hparams.predictor)

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cross_entrophy = torch.nn.CrossEntropyLoss()

    def encode(self, texts, mfccs):
        text_lengths = [i.shape[0] for i in texts]
        texts = pad_sequence(texts, batch_first=True)
        texts = self.text_encoder(texts, text_lengths)
        texts = torch.cat([texts, torch.zeros((texts.shape[0], 1, texts.shape[2]), device=texts.device)], axis=-2)

        mfcc_lengths = [i.shape[0] for i in mfccs]
        mfccs = pad_sequence(mfccs, batch_first=True)
        mfccs = self.speech_encoder(mfccs, mfcc_lengths)
        mfccs = torch.cat([mfccs, torch.zeros((mfccs.shape[0], 1, mfccs.shape[2]), device=mfccs.device)], axis=-2)

        return texts, mfccs, text_lengths, mfcc_lengths

    def positional_encoding(self, texts, mfccs):
        texts_pe = self.positional_encoding_text(texts)
        mfccs_pe = self.positional_encoding_speech(mfccs)
        return texts_pe, mfccs_pe

    def decode(self, texts, mfccs, texts_pe, mfccs_pe, text_lengths, mfcc_lengths):
        texts_at_frame, mfccs_at_text, w1, w2, score = self.attention(texts_pe, mfccs_pe, texts, mfccs, text_lengths, mfcc_lengths)

        p_texts = self.text_decoder(mfccs_at_text, text_lengths)
        p_mfccs = self.speech_decoder(texts_at_frame, mfcc_lengths)

        return p_texts, p_mfccs, w1, w2, score

    def forward(self, texts, mfccs):
        texts, mfccs, text_lengths, mfcc_lengths = self.encode(texts, mfccs)
        texts_pe, mfccs_pe = self.positional_encoding(texts, mfccs)
        p_texts, p_mfccs, w1, w2, score = self.decode(texts, mfccs, texts_pe, mfccs_pe, text_lengths, mfcc_lengths)
        boundaries = self.aligner(texts[:,:-1,:], w1, w2, text_lengths, mfcc_lengths)
        return p_texts, p_mfccs, w1, w2, boundaries

    def text_loss(self, p_texts, texts):
        p_texts = torch.cat(p_texts)
        texts = torch.cat(texts)
        return self.cross_entrophy(p_texts, texts)

    def mfcc_loss(self, p_mfccs, mfccs):
        p_mfccs = torch.cat(p_mfccs)
        mfccs = torch.cat(mfccs)
        return self.mse(p_mfccs, mfccs)

    def attention_loss(self, w1, w2, alpha=0.5):
        loss = []
        for _w1, _w2 in zip(w1, w2):
            w = torch.maximum(_w1.T, _w2)
            a = torch.linspace(1e-6, 1, w.shape[0], device=w.device).repeat(w.shape[1], 1).T
            b = torch.linspace(1e-6, 1, w.shape[1], device=w.device).repeat(w.shape[0], 1)
            r1 = torch.maximum((a / b), (b / a))
            r2 = torch.maximum(a.flip(1) / b.flip(0), b.flip(0)/ a.flip(1))
            r = torch.maximum(r1, r2) - 1
            r = torch.tanh(alpha * r)
            loss.append(torch.mean(w * r.detach()))
        loss = torch.stack(loss)
        return torch.mean(loss)

    def boundary_mae(self, p_boundaries, boundaries):
        boundaries = [i.reshape((-1,1)) for i in boundaries]
        p_boundaries = [i.reshape((-1,1)) for i in p_boundaries]
        p_boundaries = [p_boundary[boundaries[i]>-1] for i, p_boundary in enumerate(p_boundaries)]
        boundaries = [i[i>-1] for i in boundaries]
        boundaries = torch.cat(boundaries)
        p_boundaries = torch.cat(p_boundaries)
        #print(torch.median(torch.abs(p_boundaries - boundaries)))
        return self.mae(p_boundaries, boundaries)

    def extract_boundary(self, p_boundaries, threshold=0.5):
        result = []
        for p_boundary in p_boundaries:
            result.append([])
            result[-1].append(torch.FloatTensor([i[i<threshold].shape[0] / 100 for i in p_boundary[:,0,:]]))
            result[-1].append(torch.FloatTensor([i[i<threshold].shape[0] / 100 for i in p_boundary[:,1,:]]))
            #result[-1].append(torch.FloatTensor([i[i>threshold].shape[0] / 100 for i in p_boundary[:,1,:]]))
            result[-1] = torch.stack(result[-1], dim=-1).to(p_boundaries[0].device)
        return result

    def boundary_loss(self, p_boundaries, boundaries):
        boundaries = [i.reshape((-1, 1)) for i in boundaries]
        p_boundaries = [i.reshape((-1, 1, i.shape[2])) for i in p_boundaries]
        p_boundaries = [p_boundary[boundaries[i]>-1] for i, p_boundary in enumerate(p_boundaries)]
        boundaries = [i[i>-1] for i in boundaries]
        gated_boundaries = [torch.zeros(i.shape, device=p_boundaries[0].device) for i in p_boundaries]
        for i, boundary in enumerate(boundaries):
            for j, b in enumerate(boundary):
                #if j == 0:
                    gated_boundaries[i][j, int(100 * b):] = 1
                #else:
                #    gated_boundaries[i][j, :int(100 * b)] = 1
        boundaries = [i.reshape((-1,1)) for i in gated_boundaries]
        p_boundaries = [i.reshape((-1,1)) for i in p_boundaries]
        boundaries = torch.cat(boundaries)
        p_boundaries = torch.cat(p_boundaries)
        return self.mae(p_boundaries, boundaries)

class NeuFA_TeP(NeuFA_base):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.tep = nn.Linear(hparams.speech_encoder.output_dim, 1)

    def positional_encoding(self, texts, mfccs):
        tep = torch.relu(self.tep(mfccs)).squeeze(-1)
        tep = torch.cumsum(tep, dim=-1)

        texts_pe = self.positional_encoding_text(texts)
        mfccs_pe = self.positional_encoding_speech(mfccs, tep)
        return texts_pe, mfccs_pe, tep

    def forward(self, texts, mfccs):
        texts, mfccs, text_lengths, mfcc_lengths = self.encode(texts, mfccs)
        texts_pe, mfccs_pe, tep = self.positional_encoding(texts, mfccs)
        p_texts, p_mfccs, w1, w2, score = self.decode(texts, mfccs, texts_pe, mfccs_pe, text_lengths, mfcc_lengths)
        boundaries = self.aligner(texts[:,:-1,:], w1, w2, text_lengths, mfcc_lengths)
        p_text_lengths = [tep[i][l-1] for i, l in enumerate(mfcc_lengths)]
        return p_texts, p_mfccs, w1, w2, text_lengths, p_text_lengths, boundaries

    def length_loss(self, lengths, p_lengths, normalize=True):
        p_lengths = torch.stack(p_lengths)
        lengths = torch.FloatTensor(lengths).to(p_lengths.device)
        if normalize:
            p_lengths = p_lengths / lengths.detach()
            lengths = lengths / lengths.detach()
            return self.mae(lengths, p_lengths)
        else:
            return self.mse(lengths, p_lengths)

class NeuFA_MeP(NeuFA_base):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.mep = nn.Linear(hparams.text_encoder.output_dim, 1)

    def positional_encoding(self, texts, mfccs):
        mep = torch.relu(self.mep(texts)).squeeze(-1)
        mep = torch.cumsum(mep, dim=-1)

        texts_pe = self.positional_encoding_text(texts, mep)
        mfccs_pe = self.positional_encoding_speech(mfccs)
        return texts_pe, mfccs_pe, mep

    def forward(self, texts, mfccs):
        texts, mfccs, text_lengths, mfcc_lengths = self.encode(texts, mfccs)
        texts_pe, mfccs_pe, mep = self.positional_encoding(texts, mfccs)
        p_texts, p_mfccs, w1, w2, score = self.decode(texts, mfccs, texts_pe, mfccs_pe, text_lengths, mfcc_lengths)
        boundaries = self.aligner(texts[:,:-1,:], w1, w2, text_lengths, mfcc_lengths)
        p_mfcc_lengths = [mep[i][l-1] for i, l in enumerate(text_lengths)]
        return p_texts, p_mfccs, w1, w2, mfcc_lengths, p_mfcc_lengths, boundaries

    length_loss = NeuFA_TeP.length_loss

class NeuFA_TeMP(NeuFA_TeP, NeuFA_MeP):

    def positional_encoding(self, texts, mfccs):
        tep = torch.relu(self.tep(mfccs)).squeeze(-1)
        tep = torch.cumsum(tep, dim=-1)

        mep = 10 * torch.relu(self.mep(texts)).squeeze(-1)
        mep = torch.cumsum(mep, dim=-1)

        texts_pe1 = self.positional_encoding_text(texts)
        mfccs_pe1 = self.positional_encoding_speech(mfccs)
        texts_pe2 = self.positional_encoding_text(texts, mep)
        mfccs_pe2 = self.positional_encoding_speech(mfccs, tep)
        texts_pe = torch.cat([texts_pe1, texts_pe2], dim=-1)
        mfccs_pe = torch.cat([mfccs_pe2, mfccs_pe1], dim=-1)
        return texts_pe, mfccs_pe, tep, mep

    def forward(self, texts, mfccs):
        texts, mfccs, text_lengths, mfcc_lengths = self.encode(texts, mfccs)
        texts_pe, mfccs_pe, tep, mep = self.positional_encoding(texts, mfccs)
        p_texts, p_mfccs, w1, w2, score = self.decode(texts, mfccs, texts_pe, mfccs_pe, text_lengths, mfcc_lengths)
        boundaries = self.aligner(texts[:,:-1,:], w1, w2, text_lengths, mfcc_lengths)
        p_text_lengths = [tep[i][l-1] for i, l in enumerate(mfcc_lengths)]
        p_mfcc_lengths = [mep[i][l-1] for i, l in enumerate(text_lengths)]
        return p_texts, p_mfccs, w1, w2, text_lengths, p_text_lengths, mfcc_lengths, p_mfcc_lengths, boundaries

if __name__ == '__main__':
    import os
    from data.buckeye import Buckeye
    from data.common import Collate
    from hparams import base, temp

    device = 'cuda:5'
    batch_size = 4

    dataset = Buckeye(os.path.expanduser('~/BuckeyeTrain'), reduction=base.reduction_rate)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)

    for batch in data_loader:
        for Model in [NeuFA_TeMP, NeuFA_MeP, NeuFA_TeP, NeuFA_base]:
            if Model == NeuFA_TeMP:
                model = Model(temp)
            else:
                model = Model(base)
            model.to(device)
            output = model(*batch[:2])
            #print([i.shape for i in output])
            print(model.text_loss(output[0], batch[0]))
            print(model.mfcc_loss(output[1], batch[1]))
            print(model.boundary_loss(output[-1], batch[2]))
            print(model.boundary_mae(model.extract_boundary(output[-1]), batch[2]))
            print(model.attention_loss(output[2], output[3]))
            if Model in [NeuFA_TeP, NeuFA_MeP, NeuFA_TeMP]:
                print(model.length_loss(output[4], output[5]))
            if Model in [NeuFA_TeMP]:
                print(model.length_loss(output[6], output[7]))
            break
        break
