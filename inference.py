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

    def fit_to_word(self, matrix, words):
        phonemes = self.get_phonemes(words)

        result = []
        start = 0
        for word, phoneme in zip(words, phonemes):
            if len(phoneme) > 0:
                result.append(np.mean(matrix[start:start+len(phoneme)], axis=0, keepdims=True))
                start += len(phoneme)
            else:
                result.append(np.zeros((1, matrix.shape[-1])))
        result = np.concatenate(result)
        return result

    def get_words(self, text):
        if os.path.exists(text):
            with open(text) as f:
                text = f.readline().strip('\r\n').lower()
        text = ''.join([i for i in text if i in "abcedfghijklmnopqrstuvwxyz' "])
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
        return torch.IntTensor(phonemes).to(self.device)

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
        return torch.FloatTensor(mfcc).to(self.device)

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
        return boundaries[0], w1[0].cpu().numpy(), w2[0].cpu().numpy()

if __name__ == '__main__':
    import sys
    from pathlib import Path
    from functools import partial
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int, help='The GPU to use. Default is using CPU.')
    parser.add_argument('-m', '--load_model', default='neufa.pt', help='Path to exported NeuFA model. Default is neufa.pt.')
    parser.add_argument('-t', '--input_text', default=None, help='Path to the text to align. Will be ignored when processing a folder.')
    parser.add_argument('-w', '--input_wav', default=None, help='Path to the wave to align. Will be ignored when processing a folder.')
    parser.add_argument('-d', '--input_folder', default=None, help='Path of a folder containing both the text and wave files to align.')
    args = parser.parse_args()

    if args.gpu < 0:
        neufa = NeuFA()
    else:
        neufa = NeuFA(device=f'cuda:{args.gpu}')

    if args.input_folder:
        texts = [i for i in Path(args.input_folder).rglob('*.txt')]
        for text in tqdm(texts):
            wav = text.parent / f'{text.stem}.wav'
            words = neufa.get_words(text)
            boundaries, w_tts, w_asr = neufa.align(text, wav)
            #np.save(text.parent / f'{text.stem}.boundary.npy', boundaries)
            np.save(text.parent / f'{text.stem}.wasr.npy', neufa.fit_to_word(w_asr, words))
            #np.save(text.parent / f'{text.stem}.wtts.npy', w_tts)
    else:
        boundaries, w1, w2 = neufa.align(args.input_text, args.input_wav)
        words = neufa.get_words(args.input_text)
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
