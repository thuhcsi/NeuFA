## NeuFA: Neural network-based forced alignment with bidirectional attention mechanism

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
