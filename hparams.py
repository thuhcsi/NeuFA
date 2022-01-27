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
base.reduction_rate = 1
base.strategy = 'pretrain'

base.input.num_symbols = 84 + 1
base.input.max_frames = 4500 // base.reduction_rate
base.input.mfcc_dim = 39
base.input.mfcc_dim *= base.reduction_rate

base.text_encoder.num_symbols = base.input.num_symbols
base.text_encoder.embedding_dim = 256
base.text_encoder.prenet.sizes = [256, 128]
base.text_encoder.cbhg.dim = 128
base.text_encoder.cbhg.K = 16
base.text_encoder.cbhg.projections = [128, 128]
base.text_encoder.output_dim = base.text_encoder.cbhg.dim * 2
base.text_encoder.cnn.kernel_size = 5
base.text_encoder.cnn.num_layers = 3

base.speech_encoder.input_dim = base.input.mfcc_dim
base.speech_encoder.cnn.kernel_size = 1 + 2 * int(8/base.reduction_rate)
base.speech_encoder.cnn.filters = [512, 512, 512]
#base.speech_encoder.filters = [32, 32, 64, 64, 128, 128]
#base.speech_encoder.stride = (1, 1)
#base.speech_encoder.kernel_size = (3, 3)
base.speech_encoder.gru_dim = 256
base.speech_encoder.output_dim = base.speech_encoder.gru_dim * 2

base.attention.dim = 128
base.attention.text_input_dim = base.text_encoder.output_dim
base.attention.speech_input_dim = base.speech_encoder.output_dim
base.attention.text_output_dim = base.attention.speech_input_dim
base.attention.speech_output_dim = base.attention.text_input_dim

base.text_decoder.input_dim = base.attention.text_output_dim
base.text_decoder.lstm_dim = 128
base.text_decoder.output_dim = base.input.num_symbols

base.speech_decoder.input_dim = base.attention.speech_output_dim
base.speech_decoder.lstm_dim = 256
base.speech_decoder.output_dim = base.input.mfcc_dim

base.aligner.input_dim = 6
base.aligner.lstm_dim = 32
base.aligner.max_frames = base.input.max_frames
base.aligner.location_layer.attention_n_filters = 32
base.aligner.location_layer.attention_kernel_size = 31
base.aligner.location_layer.output_dim = base.aligner.location_layer.attention_n_filters
base.aligner.cnn.kernel_size = (1 + 2 * int(8/base.reduction_rate), 1 + 2 * int(8/base.reduction_rate))
base.aligner.cnn.filters = [32, 32, 32]
#base.aligner.cnn.kernel_size = (1 + 2 * int(8/base.reduction_rate), 1 + 2 * int(32/base.reduction_rate))
#base.aligner.cnn.filters = [64, 64, 64]

base.predictor.max_frames = 150
base.predictor.input_dim = (base.predictor.max_frames * 2 + 2) * 2 + base.text_encoder.output_dim
#base.predictor.input_dim = (base.predictor.max_frames * 2 + 2) * 2
base.predictor.lstm_dim = 256

base.speech_loss = 1
base.text_loss = 0.1
base.tep_loss = 10
base.mep_loss = 10
base.attention_loss = 1e3
base.attention_loss_alpha = 0.5
base.boundary_loss = 100

temp = deepcopy(base)
temp.attention.text_input_dim = temp.text_encoder.output_dim * 2
temp.attention.speech_input_dim = temp.speech_encoder.output_dim * 2
temp.attention.text_output_dim = temp.attention.speech_input_dim
temp.attention.speech_output_dim = temp.attention.text_input_dim

test = deepcopy(base)
test.batch_size = 32
test.learning_rate = 1e-3
test.text_decoder.input_dim = base.text_encoder.output_dim
test.text_encoder.embedding_dim = 256

test2 = deepcopy(test)
test2.batch_size = 256
