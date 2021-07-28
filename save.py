import os
import torch
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

from tornado.log import enable_pretty_logging
from tornado.options import options
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    #xlabel = 'Decoder timestep'
    #if info is not None:
    #    xlabel += '\n\n' + info
    #plt.xlabel(xlabel)
    #plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

class Save:

    def __init__(self, name='noname'):

        self.name = name + datetime.now().strftime("-%Y%m%d-%H%M%S")
        self.path = os.path.join('save', self.name)
        Path(self.path).mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.name)
        options.logging = 'debug'
        enable_pretty_logging(options=options, logger=self.logger)

        self.writer = SummaryWriter(self.path)

    def training_log(self, epoch, batch, step, loss):
        self.logger.info('epoch %d batch %d step %d loss %f', epoch, batch, step, loss)
        self.writer.add_scalar("training loss", loss, step)

    def validation_log(self, epoch, step, loss):
        self.logger.info('Validation epoch %d step %d loss %f', epoch, step, loss)
        self.writer.add_scalar("validation loss", loss, epoch)

    def save_model(self, model, filename):
        torch.save(model.state_dict(), os.path.join(self.path, filename))

    def training_attention(self, step, w1, w2):
        self.writer.add_image("w1", plot_alignment_to_numpy(w1.data.cpu().numpy()), step, dataformats='HWC')
        self.writer.add_image("w2", plot_alignment_to_numpy(w2.data.cpu().numpy()), step, dataformats='HWC')

if __name__ == '__main__':
    save = Save()
    save.logger.info('Test')
