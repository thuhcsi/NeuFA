## NeuFA: Neural network-based forced alignment with bidirectional attention mechanism

### Please read this first

> Well, it would be a superise to me if NeuFA (or any other FA model) predicts some insane boundaries.
>
> Like the paper said, the 50 ms tolerance accuracy of NeuFA is 95% at word level.
It seems to be high. But in practice, for a sentence with 20 phonemes in example. The possibilty that there is a phoneme with a predicted boundary 50ms biased from the ground-truth is `1 - .95 ^ 20 = 64.15%`. Similarly, the possibilty that there is a phoneme with a predicted boundary 100ms biased from the ground-truth is `1 - .98 ^ 20 = 33.24%`.
>
> Also, NeuFA currently doesn't restrict the predicted boundaries to be nonoverlapping (we are working on this in NeuFA 2),
which makes the situation even worse.
>
> So my opinion is NeuFA is not ready for production enviroments yet.
But NeuFA could be used as a "soft" FA model which extracts the attention weights between the text and speech to map the information between them. And this is exactly why we propose NeuFA and how we use it in our other researches.

### Usage

* Clone this repository and its submodules.

```shell
git clone https://github.com/thuhcsi/NeuFA
cd NeuFA
git submodule update --init --recursive
```

* Download and decompress the [LibriSpeech](https://www.openslr.org/12) and [Buckeye](https://buckeyecorpus.osu.edu/) corpus.

* Run this command to preprocess the LibriSpeech corpus.

```shell
python -m data.librispeech /path/to/LibriSpeech
```

* Run this command **twice** to preprocess the Buckeye corpus.

```shell
python -m data.buckeye /path/to/LibriSpeech
python -m data.buckeye /path/to/LibriSpeech
```
* Split the training and test sets.

```shell
mkdir -p /path/to/Buckeye{Train,Test}/segmented
mv /path/to/Buckeye/segmented/s{10,20,30,40}* /path/to/BuckeyeTest/segmented
mv /path/to/Buckeye/segmented/* /path/to/BuckeyeTrain/segmented
```

* Set the training strategy in `hparams.py` as one of `pretrain`, `finetune` and `semi`.

```python
base.strategy = 'semi'
```

* Train.

```shell
python train.py --gpu 0 --train_path /path/to/LibriSpeech --dev_path /path/to/BuckeyeTrain --valid_path /path/to/BuckeyeTest
```
The training logs and models are saved in `save`. Also support TensorBoard.

* Inference.

Export a saved model.
```shell
python misc/export.py /path/to/checkpoint neufa.pt
```

Inference with given text and wave files.
```shell
python inference.py -m neufa.pt -t /path/to/text.txt -w /path/to/wave.wav
```

Or inference with a folder containing the text and wave files.
```shell
python inference.py -m neufa.pt -d /path/to/folder
```
