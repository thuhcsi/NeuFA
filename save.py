import os
import json
import logging
import torch
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
        self.path = Path('save') / self.name
        self.path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.name)
        options.logging = 'debug'
        enable_pretty_logging(options=options, logger=self.logger)

        self.writer = SummaryWriter(self.path)

    def save_log(self, stage, epoch, batch, step, loss):
        self.logger.info('[%s] %s epoch %d batch %d step %d loss %f', self.name, stage, epoch, batch, step, loss)
        self.writer.add_scalar(f"{stage}/loss", loss, step)

    def save_parameters(self, hparams):
        self.writer.add_text("hparams", json.dumps(hparams, indent=2))

    def save_model(self, model, filename):
        torch.save(model.state_dict(), os.path.join(self.path, filename))

    def save_boundary(self, stage, step, p_boundary, boundary, shape):
        figure = np.zeros(shape)

        for i in range(boundary.shape[0]):
            for j, k in [(p_boundary[i][0], 0.7), (p_boundary[i][1], 0.7), (boundary[i][0], 1), (boundary[i][1], 1)]:
                try:
                    if j >= 0:
                        #print(int(100 * j), i, k)
                        figure[int(100 * j), i] = k
                except:
                    pass

        self.writer.add_image(f"{stage}/boundary", plot_alignment_to_numpy(figure), step, dataformats='HWC')

    def save_attention(self, stage, step, w1, w2):
        self.writer.add_image(f"{stage}/w1", plot_alignment_to_numpy(w1.T.data.cpu().numpy()), step, dataformats='HWC')
        self.writer.add_image(f"{stage}/w2", plot_alignment_to_numpy(w2.T.data.cpu().numpy()), step, dataformats='HWC')

if __name__ == '__main__':
    save = Save()
    save.logger.info('Test')
