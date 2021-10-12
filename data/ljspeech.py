import os
import sys
import numpy as np
import librosa
import unicodedata
import re
import torch
from pathlib import Path
from functools import partial

punctuation_re = re.compile(r'[.,?!’"“”:;-]')
comment_re = re.compile(r'\[[^]]*\]')

def process_text(path, g2p, line):
    line = line.strip('\r\n').split('|')
    text, _ = punctuation_re.subn(' ', line[-1].lower())
    text, _ = comment_re.subn(' ', text)
    text = ''.join([i for i in unicodedata.normalize('NFKD', text) if not unicodedata.combining(i)])
    words = text.split(' ')
    words = [i for i in words if i != '']
    phonemes = []
    for word in words:
        phonemes += g2p.convert(word)
    phonemes = [i[:-1] if i.endswith(('0', '1', '2')) else i for i in phonemes]
    phonemes = [g2p.symbol2id[i] + 1 for i in phonemes if i in g2p.symbols]
    phonemes = np.array(phonemes)
    np.save(path / 'wavs' / (line[0] + '.text.npy'), phonemes)

def process_wav(file: Path, sample_rate=22050):
    waveform, _ = librosa.load(file, sr=sample_rate, mono=True)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13, hop_length=int(sample_rate/100), n_fft=int(sample_rate/40), fmax=8000)
    delta = librosa.feature.delta(mfcc, width=3, order=1)
    delta2 = librosa.feature.delta(mfcc, width=3, order=2)
    np.save(file.parent / (file.name[:-3] + 'mfcc.npy'), np.concatenate([mfcc, delta, delta2]).T.astype(np.float32))

def get_mean_and_std(mfccs):
    mfccs = np.concatenate(mfccs, axis=0)
    mean = mfccs.mean(axis=0, keepdims=False)
    std = mfccs.std(axis=0, keepdims=False)
    return mean, std

def normalize(mean, standard, inputs):
    file, mfcc = inputs
    mfcc -= mean
    mfcc /= standard
    np.save(file.parent / (file.name[:-3] + 'normalized.mfcc.npy'), mfcc.astype(np.float32))

class LJSpeech(torch.utils.data.Dataset):

    def __init__(self, path, reduction: int = 1):
        super().__init__()
        self.path = Path(path)
        self.wavs = [i for i in self.path.rglob('*.wav')]
        self.texts = [Path(str(i)[:-3] + 'text.npy') for i in self.wavs]
        self.mfccs = [Path(str(i)[:-3] + 'mfcc.npy') for i in self.wavs]
        self.normalized_mfccs = [Path(str(i)[:-3] + 'normalized.mfcc.npy') for i in self.wavs]
        self.reduction = reduction

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, index):
        text = np.load(self.texts[index])
        mfcc = np.load(self.normalized_mfccs[index])

        if mfcc.shape[0] % self.reduction != 0:
            mfcc = np.concatenate([mfcc, np.zeros((self.reduction - mfcc.shape[0] % self.reduction, mfcc.shape[1]))])
        if self.reduction > 1:
            mfcc = mfcc.reshape(mfcc.shape[0] // self.reduction, mfcc.shape[1] * self.reduction)

        return text, mfcc.astype(np.float32)

if __name__ == '__main__':
    from tqdm.contrib.concurrent import process_map, thread_map
    dataset = LJSpeech(sys.argv[1], reduction=4)
    lines = open(dataset.path / 'metadata.csv').readlines()

    if not dataset.texts[0].exists():
        from g2p.en_us import G2P
        g2p = G2P()
        thread_map(partial(process_text, dataset.path, g2p), lines, chunksize=1)

    if not dataset.mfccs[0].exists():
        thread_map(process_wav, dataset.wavs)

    if not dataset.normalized_mfccs[0].exists():
        mfccs = thread_map(np.load, dataset.mfccs)
        mean, std = get_mean_and_std(mfccs)
        thread_map(partial(normalize, mean, std), [i for i in zip(dataset.wavs, mfccs)])

    from .common import Collate
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=Collate('cuda:0'), drop_last=True)
    for batch in data_loader:
        for _list in batch:
            print([i.shape for i in _list])
            print(_list[1])
        break
