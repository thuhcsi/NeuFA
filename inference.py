import os
import torch
import numpy as np
import librosa
from g2p.en_us import G2P

class NeuFA:

    def __init__(self, model_path='neufa.pt', device='cpu'):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.g2p = G2P()

    def get_words(self, text):
        if os.path.exists(text):
            with open(text) as f:
                text = f.readline().strip('\r\n').lower()
        text = ''.join([i for i in text if i in 'abcdefghijklmnopqrstuvwxyz '])
        words = text.split(' ')
        return words

    def get_phonemes(self, words):
        phonemes = []
        for word in words:
            phonemes += [self.g2p.convert(word)]
        for j, phoneme in enumerate(phonemes):
            phonemes[j] = [i[:-1] if i.endswith(('0', '1', '2')) else i for i in phoneme]
            phonemes[j] = [self.g2p.symbol2id[i] + 1 for i in phoneme if i in self.g2p.symbols]
        return phonemes

    def load_text(self, text):
        words = self.get_words(text)
        phonemes = self.get_phonemes(words)
        phonemes = [j for i in phonemes for j in i]
        phonemes = np.array(phonemes)
        return torch.IntTensor(phonemes, device=self.device)

    def load_wav(self, wav):
        if os.path.exists(wav):
            wav, sample_rate = librosa.load(wav, mono=True)
        mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13, hop_length=int(sample_rate/100), n_fft=int(sample_rate/40), fmax=8000)
        delta = librosa.feature.delta(mfcc, width=3, order=1)
        delta2 = librosa.feature.delta(mfcc, width=3, order=2)
        mfcc = np.concatenate([mfcc, delta, delta2]).T.astype(np.float32)
        mean = mfcc.mean(axis=0, keepdims=False)
        std = mfcc.std(axis=0, keepdims=False)
        mfcc -= mean
        mfcc /= std
        return torch.FloatTensor(mfcc, device=self.device)

    def extract_boundary(self, p_boundaries, threshold=0.5):
        result = []
        for p_boundary in p_boundaries:
            result.append([])
            result[-1].append(np.array([i[i<threshold].shape[0] / 100 for i in p_boundary[:,0,:]]))
            result[-1].append(np.array([i[i<threshold].shape[0] / 100 for i in p_boundary[:,1,:]]))
            result[-1] = np.stack(result[-1], axis=-1)
        return result

    def align(self, text, wav):
        text = [self.load_text(text)]
        wav = [self.load_wav(wav)]
        with torch.no_grad():
            _, _, w1, w2, _, _, _, _, boundaries = self.model(text, wav)
            boundaries = self.extract_boundary(boundaries)
        return boundaries[0], w1[0].numpy(), w2[0].numpy()

if __name__ == '__main__':
    import sys

    neufa = NeuFA()
    boundaries, w1, w2 = neufa.align(sys.argv[1], sys.argv[2])
    words = neufa.get_words(sys.argv[1])
    phonemes = neufa.get_phonemes(words)
    start = 0
    for word, phoneme in zip(words, phonemes):
        if len(phoneme) > 0:
            #l = np.min(boundaries[start:start+len(phoneme)])
            #r = np.max(boundaries[start:start+len(phoneme)])
            l = boundaries[start, 0]
            r = boundaries[start+len(phoneme) - 1, 1]
            t = r - l
            print(word, l, r, '%.2f' % t)
        else:
            print(word, '-', '-')

        for p, boundary in zip(phoneme, boundaries[start:start+len(phoneme)]):
            print(neufa.g2p.id2symbol[p-1], boundary)

        start += len(phoneme)
