from copy import deepcopy

class Options(dict):

    def __getitem__(self, key):
        if not key in self.keys():
            self.__setitem__(key, Options())
        return super().__getitem__(key)

    def __getattr__(self, attr):
        if not attr in self.keys():
            self[attr] = Options()
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]

    def __deepcopy__(self, memo=None):
        new = Options()
        for key in self.keys():
            new[key] = deepcopy(self[key])
        return new

base = Options()
base.max_epochs = 50000
base.batch_size = 16
base.learning_rate = 1e-3

base.input.num_symbols = 84 + 1
base.input.mfcc_dim = 39

base.text_encoder.num_symbols = base.input.num_symbols
base.text_encoder.embedding_dim = 256
base.text_encoder.prenet.sizes = [256, 128]
base.text_encoder.cbhg.dim = 128
base.text_encoder.cbhg.K = 16
base.text_encoder.cbhg.projections = [128, 128]
base.text_encoder.output_dim = base.text_encoder.cbhg.dim * 2

base.speech_encoder.input_dim = base.input.mfcc_dim
base.speech_encoder.filters = [32, 32, 64, 64, 128, 128]
base.speech_encoder.kernel_size = (3, 3)
base.speech_encoder.stride = (1, 1)
base.speech_encoder.padding = (1, 1)
base.speech_encoder.gru_dim = 64
base.speech_encoder.output_dim = base.speech_encoder.gru_dim * 2

base.attention.dim = 128

base.text_decoder.input_dim = base.speech_encoder.output_dim
base.text_decoder.lstm_dim = 128
base.text_decoder.output_dim = base.input.num_symbols

base.speech_decoder.input_dim = base.text_encoder.output_dim
base.speech_decoder.lstm_dim = 128
base.speech_decoder.output_dim = base.input.mfcc_dim

test = deepcopy(base)
test.batch_size = 64
test.learning_rate = 1e-3
test.text_decoder.input_dim = base.text_encoder.output_dim
test.text_encoder.cnn.kernel_size = 5
test.text_encoder.cnn.num_layers = 3

test2 = deepcopy(test)
test2.batch_size = 256
