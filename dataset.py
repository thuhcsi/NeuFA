import io
import torch
import numpy as np
import soundfile as sf
import librosa
from multiprocessing import Pool

from hdfs import HDFSDataset, HDFSLoader
from dataloader import KVReader

class Collate:

    def __init__(self, device=None):
        self.device = device

    def __call__(self, batch):
        length = len(batch[0])
        output = [[] for i in range(length)]

        for data in batch:
            for i, j in enumerate(data):
                output[i].append(j if self.device is None else j.to(self.device))

        return tuple(output)

def mfcc(waveform):
    print(waveform)
    result = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13)
    return result

class LibriSpeech(HDFSDataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.keys = KVReader(path).list_keys()
        self.wav_reader = None
        self.pool = None

    def get_items(self, indices: list, num_readers=1):
        if self.wav_reader is None or self.wav_reader.num_reader != num_readers:
            self.wav_reader = KVReader(self.path, num_readers)
            self.pool = Pool(num_readers)

        keys = [self.keys[i] for i in indices]
        byte_waveforms = self.wav_reader.read_many(keys)

        waveforms = []
        for b_wav in byte_waveforms:
            audio_data, _ = sf.read(io.BytesIO(b_wav), always_2d=True)
            if audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=0, keepdims=True)
            audio_data = audio_data.squeeze()
            waveforms.append(audio_data)

        mfccs = self.pool.map(mfcc, waveforms)
        mfccs = torch.from_

        return waveforms, mfccs

if __name__ == '__main__':

    dataset = LibriSpeech('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/dev-clean')
    batch_size = 2
    data_loader = HDFSLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Collate('cuda:0'), drop_last=True, num_readers=batch_size)
    for batch in data_loader:
        print(batch)
        break
