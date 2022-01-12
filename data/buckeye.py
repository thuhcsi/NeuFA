import os
import sys
import subprocess
import numpy as np
import librosa
import torch
from pathlib import Path
from functools import partial

phoneme_map = {
        'a': 'AH',
        'e': 'EH',
        'h': 'HH',
        'i': 'IH',

        'tq': 'T',
        'q': 'TH',
        'id': 'D',

        'dx': ['D', 'Z'],
        'nx': ['N', 'Z'],
        'aan': ['AA', 'N'],
        'aen': ['AE', 'N'],
        'ahn': ['AH', 'N'],
        'aon': ['AO', 'N'],
        'awn': ['AW', 'N'],
        'ayn': ['AY', 'N'],
        'ehn': ['EH', 'N'],
        'ern': ['ER', 'N'],
        'eyn': ['EY', 'N'],
        'hhn': ['HH', 'N'],
        'ihn': ['IH', 'N'],
        'iyn': ['IY', 'N'],
        'own': ['OW', 'N'],
        'oyn': ['OY', 'N'],
        'uhn': ['UH', 'N'],
        'uwn': ['UW', 'N'],

        'ah l': ['AH', 'L'],
        'ah n': ['AH', 'N'],
        'ah r': ['AH', 'R'],
        'ih l': ['IH', 'L'],

        'en': ['AE', 'N'],
        'em': ['AH', 'M'],
        'el': ['AH', 'L'],
        'eng': ['IH', 'NG'],
}

class Word:

    def __init__(self, content=None, start=None, end=None):
        self.content = content
        self.start = start
        self.end = end

class Phoneme(Word):
    pass

class Sentence:

    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end
        self.words = [Word(start=start)]

    def __str__(self):
        return str([word.content for word in self.words])

def clip_wav(source, start, end, target):
    duration = '%.6f' % (end - start)
    start = '%.6f' % start
    subprocess.run(['ffmpeg', '-i', source, '-ss', start, '-t', duration, target, '-v', 'quiet', '-y'])

def process_segmentation(segmented, g2p, path):
    lines = open(path).readlines()[9:]
    lines = [line.strip('\r\n') for line in lines]

    sentences = []
    for line in lines:
        line = line.split(';')[0].split(' ')
        line = [i for i in line if i != '']
        if len(line) == 0:
            continue
        if len(line) > 3:
            line[2] = ' '.join(line[2:])
            del line[3:]
            if not line[2].startswith('<') or not line[-1].endswith('>'):
                print('Invaild word:', path, line[2])
        word = line[2]
        time = float(line[0])
        if word in ['{B_TRANS}', '<IVER>', '{E_TRANS}', '<VOCNOISE>', '<SIL>']:
            if len(sentences) > 0 and sentences[-1].end is None:
                sentences[-1].end = sentences[-1].words[-1].end
            sentences.append(Sentence(start=time))
        else:
            if len(sentences) == 0:
                sentences.append(Sentence(start=0))
            if sentences[-1].words[-1].end is None:
                sentences[-1].words[-1].end = time
                sentences[-1].words[-1].content = word
            else:
                sentences[-1].words.append(Word(content=word, start=sentences[-1].words[-1].end, end=time))

    lines = open(path.parent/ (path.stem + '.phones')).readlines()[9:]
    lines = [line.strip('\r\n') for line in lines]

    phonemes = []
    start = 0
    for line in lines:
        line = line.split(';')[0].split(' ')
        line = [i for i in line if i != '']
        if len(line) == 0:
            continue
        if len(line) > 3:
            line[2] = ' '.join(line[2:])
            del line[3:]
            if not line[2].startswith('<') or not line[-1].endswith('>'):
                print('Invaild phoneme:', path.stem + '.phones', line[2])
        time = float(line[0])
        if len(line) < 3:
            if phonemes[-1].end is None:
                phonemes[-1].end = time
            continue
        phoneme = line[2].replace('+1', '')
        if len(phonemes) == 0:
            phonemes.append(Phoneme(content=phoneme, start=0, end=time))
        else:
            phonemes.append(Phoneme(content=phoneme, start=phonemes[-1].end, end=time))

    for i, sentence in enumerate(sentences):
        try:
            sentence.words = [word for word in sentence.words if not word.content.startswith('<')]
        except:
            continue

        if len(sentence.words) == 0:
            continue

        if sentence.start is None or sentence.end is None:
            print('Ignored invalid sentence:', sentence)
            continue

        if sentence.end - sentence.start <= 0.025:
        #if sentence.end - sentence.start <= 0.1:
            continue

        clip_wav(path.parent / (path.stem + '.wav'), sentence.start, sentence.end, segmented / ('%s-%03d.wav' % (path.stem, i)))

        #with open(segmented / ('%s-%03d.txt' % (path.stem, i)), 'w') as f:
        #    f.writelines(' '.join([word.content for word in sentence.words]))

        lines = []

        for word in sentence.words:
            lines.append('%.6f\t%.6f\t%s' % (word.start - sentence.start, word.end - sentence.start, word.content))

        lines = [line + '\n' for line in lines]
        with open(segmented / ('%s-%03d.words' % (path.stem, i)), 'w') as f:
            f.writelines(lines)

        #lines = []
        #for word in sentence.words:
        #    #_phonemes = [j for j in phonemes if word.start <= j.end and j.start <= word.end]
        #    _phonemes = [j for j in phonemes if word.start <= j.start and j.end <= word.end]
        #    for phoneme in _phonemes:
        #        if phoneme.content.upper() in g2p.symbols:
        #            lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, phoneme.end - sentence.start, phoneme.content.upper()))
        #        elif phoneme.content in phoneme_map.keys():
        #            if type(phoneme_map[phoneme.content]) is list:
        #                for j, p in enumerate(phoneme_map[phoneme.content]):
        #                    if j == 0:
        #                        lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, -1, p))
        #                    elif j == len(phoneme_map[phoneme.content]) - 1:
        #                        lines.append('%.6f\t%.6f\t%s' % (-1, phoneme.end - sentence.start, p))
        #                    else:
        #                        lines.append('%.6f\t%.6f\t%s' % (-1, -1, p))
        #            else:
        #                lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, phoneme.end - sentence.start, phoneme_map[phoneme.content]))
        #        elif phoneme.content in ['{B_TRANS}', '{E_TRANS}', 'VOCNOISE', 'SIL', 'IVER', 'LAUGH', '<EXCLUDE-name>', 'UNKNOWN', 'NOISE', 'IVER-LAUGH', '<exclude-Name>']:
        #            continue
        #        else:
        #            print('Unknown phoneme', path.stem, i, phoneme.content)
        #            lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, phoneme.end - sentence.start, phoneme.content))
        #    lines.append('')

        lines = []
        _phonemes = [j for j in phonemes if sentence.start <= j.start and j.end <= sentence.end]
        for phoneme in _phonemes:
            if phoneme.content.upper() in g2p.symbols:
                lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, phoneme.end - sentence.start, phoneme.content.upper()))
            elif phoneme.content in phoneme_map.keys():
                if type(phoneme_map[phoneme.content]) is list:
                    for j, p in enumerate(phoneme_map[phoneme.content]):
                        if j == 0:
                            lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, -1, p))
                        elif j == len(phoneme_map[phoneme.content]) - 1:
                            lines.append('%.6f\t%.6f\t%s' % (-1, phoneme.end - sentence.start, p))
                        else:
                            lines.append('%.6f\t%.6f\t%s' % (-1, -1, p))
                else:
                    lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, phoneme.end - sentence.start, phoneme_map[phoneme.content]))
            elif phoneme.content in ['{B_TRANS}', '{E_TRANS}', 'VOCNOISE', 'SIL', 'IVER', 'LAUGH', '<EXCLUDE-name>', 'UNKNOWN', 'NOISE', 'IVER-LAUGH', '<exclude-Name>']:
                continue
            else:
                print('Unknown phoneme', path.stem, i, phoneme.content)
                lines.append('%.6f\t%.6f\t%s' % (phoneme.start - sentence.start, phoneme.end - sentence.start, phoneme.content))

        with open(segmented / ('%s-%03d.phonemes' % (path.stem, i)), 'w') as f:
            f.write('\n'.join(lines))

def process_word(g2p, path):
    lines = open(path).readlines()
    lines = [i.strip('\r\n') for i in lines]
    lines = [i.split('\t') for i in lines]
    words = [i[2] for i in lines]

    phonemes = []
    for word in words:
        phonemes += g2p.convert(word)
    phonemes = [g2p.symbol2id[i] + 1 for i in phonemes if i in g2p.symbols]
    if len(phonemes) == 0:
        print('Removing sentence with no phoneme:', path)
        for i in path.parent.rglob(f'{path.stem}.*'):
            i.unlink()
        return

    phonemes = np.array(phonemes)
    np.save(path.parent / (path.stem + '.word.npy'), phonemes)

    result = []
    for i, word in enumerate(words):
        phonemes = g2p.convert(word)
        result.append(np.zeros((len(phonemes), 2)) - 1)
        result[-1][0, 0] = float(lines[i][0])
        result[-1][-1, 1] = float(lines[i][1])

    result = np.concatenate(result)
    np.save(path.parent / (path.stem + '.word.boundary.npy'), result)

def process_phoneme(g2p, path):
    lines = open(path).readlines()
    lines = [line.strip('\r\n') for line in lines]
    lines = [line.split('\t') for line in lines]

    phonemes = [i[2] for i in lines]
    if False in [phoneme in g2p.symbols for phoneme in phonemes]:
        print('Removing sentence with unknown phoneme:', path)
        for i in path.parent.rglob(f'{path.stem}.*'):
            i.unlink()
        return

    phonemes = [g2p.symbol2id[i] + 1 for i in phonemes if i in g2p.symbols]
    if len(phonemes) == 0:
        print('Removing sentence with no phoneme:', path)
        for i in path.parent.rglob(f'{path.stem}.*'):
            i.unlink()
        return

    phonemes = np.array(phonemes)
    np.save(path.parent / (path.stem + '.phoneme.npy'), phonemes)

    result = np.zeros((len(lines), 2))
    for i, line in enumerate(lines):
        result[i, 0] = float(line[0])
        result[i, 1] = float(line[1])

    np.save(path.parent / (path.stem + '.phoneme.boundary.npy'), result)

def process_wav(file: Path, sample_rate=16000):
    waveform, sample_rate = librosa.load(file, sr=sample_rate, mono=True)
    if len(waveform) == 0:
        print('Removing empty wav:', file)
        for i in file.parent.rglob(f'{file.stem}.*'):
            i.unlink()
        return
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13, hop_length=int(sample_rate/100), n_fft=int(sample_rate/40), fmax=8000)
    delta = librosa.feature.delta(mfcc, width=3, order=1)
    delta2 = librosa.feature.delta(mfcc, width=3, order=2)
    np.save(file.parent / (file.name[:-3] + 'mfcc.npy'), np.concatenate([mfcc, delta, delta2]).T.astype(np.float32))

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

class Buckeye(torch.utils.data.Dataset):

    def __init__(self, path, reduction: int = 1):
        super().__init__()
        self.path = Path(path)
        self.segmented = self.path / 'segmented'
        self.wavs = [i for i in self.segmented.rglob('*.wav')]
        self.words = [Path(str(i)[:-3] + 'word.npy') for i in self.wavs]
        self.phonemes = [Path(str(i)[:-3] + 'phoneme.npy') for i in self.wavs]
        self.word_boundaries = [Path(str(i)[:-3] + 'word.boundary.npy') for i in self.wavs]
        self.phoneme_boundaries = [Path(str(i)[:-3] + 'phoneme.boundary.npy') for i in self.wavs]
        self.mfccs = [Path(str(i)[:-3] + 'mfcc.npy') for i in self.wavs]
        self.normalized_mfccs = [Path(str(i)[:-3] + 'normalized.mfcc.npy') for i in self.wavs]
        self.reduction = reduction

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, index):
        word = np.load(self.words[index])
        word_boundary = np.load(self.word_boundaries[index])
        mfcc = np.load(self.normalized_mfccs[index])

        if mfcc.shape[0] % self.reduction != 0:
            mfcc = np.concatenate([mfcc, np.zeros((self.reduction - mfcc.shape[0] % self.reduction, mfcc.shape[1]))])
        if self.reduction > 1:
            mfcc = mfcc.reshape(mfcc.shape[0] // self.reduction, mfcc.shape[1] * self.reduction)

        return word, mfcc.astype(np.float32), word_boundary.astype(np.float32)

class BuckeyePhoneme(Buckeye):

    def __getitem__(self, index):
        phoneme = np.load(self.phonemes[index])
        phoneme_boundary = np.load(self.phoneme_boundaries[index])
        mfcc = np.load(self.normalized_mfccs[index])

        if mfcc.shape[0] % self.reduction != 0:
            mfcc = np.concatenate([mfcc, np.zeros((self.reduction - mfcc.shape[0] % self.reduction, mfcc.shape[1]))])
        if self.reduction > 1:
            mfcc = mfcc.reshape(mfcc.shape[0] // self.reduction, mfcc.shape[1] * self.reduction)

        return phoneme, mfcc.astype(np.float32), phoneme_boundary.astype(np.float32)

if __name__ == '__main__':
    from tqdm.contrib.concurrent import process_map, thread_map
    from g2p.en_us import G2P

    g2p = G2P()
    dataset = BuckeyePhoneme(sys.argv[1], reduction=4)

    if not dataset.segmented.exists():
        dataset.segmented.mkdir(exist_ok=True)
        words = [i for i in dataset.path.rglob('*.words')]
        thread_map(partial(process_segmentation, dataset.segmented, g2p), words)
        dataset = Buckeye(sys.argv[1], reduction=4)

    if not dataset.phonemes[0].exists():
        phonemes = [i for i in dataset.segmented.rglob('*.phonemes')]
        thread_map(partial(process_phoneme, g2p), phonemes)

    if not dataset.words[0].exists():
        texts = [i for i in dataset.segmented.rglob('*.words')]
        thread_map(partial(process_word, g2p), texts)
        print('[Important] Now please run this script again to finish the preprocessing.')
        sys.exit()

    if not dataset.mfccs[0].exists():
        thread_map(process_wav, dataset.wavs)
        dataset = Buckeye(sys.argv[1], reduction=4)

    if not dataset.normalized_mfccs[0].exists():
        mfccs_per_speaker = {}
        for i in dataset.mfccs:
            name = i.name[:3]
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
            print([i.dtype for i in _list])
        break
