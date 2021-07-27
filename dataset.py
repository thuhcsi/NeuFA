import io
import torch
import numpy as np
import soundfile as sf
import librosa
from multiprocessing import Pool

from g2p.en_us import G2P
from hdfs import HDFSDataset, HDFSLoader, HDFSCollate
from dataloader import KVReader

g2p = G2P()

def wav2mfcc(waveform):
    sample_rate = 16000
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13, hop_length=int(sample_rate/100), n_fft=int(sample_rate/40), fmax=8000)
    delta = librosa.feature.delta(mfcc, width=3, order=1)
    delta2 = librosa.feature.delta(mfcc, width=3, order=2)
    return np.concatenate([mfcc, delta, delta2]).T

def load_wav(waveform):
    waveform, _ = sf.read(io.BytesIO(waveform), always_2d=True)
    if waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=0, keepdims=True)
    waveform = waveform.squeeze()
    return waveform

def load_text(text):
    text = text.decode('utf-8').lower()
    words = text.split(' ')
    phonemes = []
    for word in words:
        phonemes += g2p.convert(word)
    phonemes = [g2p.symbol2id[i] for i in phonemes]
    phonemes = torch.nn.functional.one_hot(torch.from_numpy(np.array(phonemes)), num_classes=len(g2p.symbols))
    return phonemes

class LibriSpeech(HDFSDataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.keys = KVReader(path + '_wav').list_keys()
        self.wav_reader = None
        self.txt_reader = None
        self.pool = None

    def get_items(self, indices: list, num_readers=1):
        if self.wav_reader is None or self.wav_reader.num_reader != num_readers:
            self.wav_reader = KVReader(self.path + '_wav', num_readers)
            self.txt_reader = KVReader(self.path + '_txt', num_readers)
            self.pool = Pool(num_readers)

        keys = [self.keys[i] for i in indices]
        waveforms = self.wav_reader.read_many(keys)
        waveforms = self.pool.map(load_wav, waveforms)
        mfccs = self.pool.map(wav2mfcc, waveforms)
        texts = self.txt_reader.read_many(keys)
        phonemes = self.pool.map(load_text, texts)

        return (phonemes, mfccs)

if __name__ == '__main__':

    dataset = LibriSpeech('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    batch_size = 2
    data_loader = HDFSLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=HDFSCollate('cuda:0'), drop_last=True, num_readers=batch_size)
    for batch in data_loader:
        for _list in batch:
            print(' '.join([str(i.shape) for i in _list]))
        break
