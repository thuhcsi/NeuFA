import os
import sys
import numpy as np
import librosa
import torch
from pathlib import Path
from functools import partial

def process_text(g2p, path):
    lines = open(path).readlines()
    lines = [i.strip('\r\n').split(' ') for i in lines]
    for line in lines:
        key = line[0]
        text = ' '.join(line[1:]).lower()
        words = text.split(' ')
        phonemes = []
        for word in words:
            phonemes += g2p.convert(word)
        phonemes = [i[:-1] if i.endswith(('0', '1', '2')) else i for i in phonemes]
        phonemes = [g2p.symbol2id[i] + 1 for i in phonemes if i in g2p.symbols]
        phonemes = np.array(phonemes)
        np.save(path.parent / (key + '.text.npy'), phonemes)

def process_wav(file: Path):
    waveform, sample_rate = librosa.load(file, mono=True)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13, hop_length=int(sample_rate/100), n_fft=int(sample_rate/40), fmax=8000)
    delta = librosa.feature.delta(mfcc, width=3, order=1)
    delta2 = librosa.feature.delta(mfcc, width=3, order=2)
    np.save(file.parent / f'{file.stem}.mfcc.npy', np.concatenate([mfcc, delta, delta2]).T.astype(np.float32))

def get_mean_and_std(mfccs):
    mfccs = np.concatenate(mfccs, axis=0)
    mean = mfccs.mean(axis=0, keepdims=False)
    std = mfccs.std(axis=0, keepdims=False)
    return mean, std

def process_normalization(mfccs_per_speaker):
    mfccs = [np.load(i) for i in mfccs_per_speaker]
    mean, std = get_mean_and_std(mfccs)
    for i, mfcc in enumerate(mfccs):
        mfcc -= mean
        mfcc /= std
        np.save(mfccs_per_speaker[i].parent / (mfccs_per_speaker[i].name[:-8] + 'normalized.mfcc.npy'), mfcc.astype(np.float32))

class LibriSpeech(torch.utils.data.Dataset):

    def __init__(self, path, reduction: int = 1):
        super().__init__()
        self.path = Path(path)
        self.wavs = [i for i in self.path.rglob('*.flac')]
        self.texts = [Path(str(i)[:-4] + 'text.npy') for i in self.wavs]
        self.mfccs = [Path(str(i)[:-4] + 'mfcc.npy') for i in self.wavs]
        self.normalized_mfccs = [Path(str(i)[:-4] + 'normalized.mfcc.npy') for i in self.wavs]
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
    dataset = LibriSpeech(sys.argv[1], reduction=4)

    if not dataset.texts[0].exists():
        from g2p.en_us import G2P
        g2p = G2P()
        texts = [i for i in dataset.path.rglob('*.trans.txt')]
        thread_map(partial(process_text, g2p), texts)

    if not dataset.mfccs[0].exists():
        thread_map(process_wav, dataset.wavs)

    if not dataset.normalized_mfccs[0].exists():
        mfccs_per_speaker = {}
        for i in dataset.mfccs:
            name = i.parents[1].name
            if not name in mfccs_per_speaker:
                mfccs_per_speaker[name] = [i]
            else:
                mfccs_per_speaker[name].append(i)

        thread_map(process_normalization, mfccs_per_speaker.values())

    from .common import Collate
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=Collate('cuda:0'), drop_last=True)
    for batch in data_loader:
        for _list in batch:
            print([i.shape for i in _list])
        break
