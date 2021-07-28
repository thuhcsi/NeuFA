import os
import torch
import argparse
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from dataset import LibriSpeech, LibriSpeechText
from hdfs import HDFSLoader, HDFSCollate
from save import Save

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', default=None)
parser.add_argument('--restore_path', default=None)
parser.add_argument('--model', required=True, choices=['base', 'test', 'proposed2', 'proposed3'])
args = parser.parse_args()

device = "cuda:%d" % args.gpu

if args.model == 'base':
    from model.neufa import NeuFA_base as Model
    from hparams import base as hparams
elif args.model == 'test':
    from model.test import BidirectionalAttentionTest as Model
    from hparams import test as hparams

#train_dataset = LibriSpeech('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
train_dataset = LibriSpeechText(os.path.expanduser('~/datasets/LibriSpeech/packed/LibriSpeech'))
train_dataloader = HDFSLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=HDFSCollate(device), drop_last=True, num_readers=hparams.batch_size)
#valid_dataset = dev_data(args.input_path)
#valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), collate_fn=Collate(device), drop_last=True)

if args.restore_path is not None:
    model.load_state_dict(torch.load(args.restore_path))

model = Model(hparams)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if args.name is None:
    args.name = args.model
save = Save(args.name)

step = 1
for epoch in range(hparams.max_epochs):
    save.logger.info('Epoch %d', epoch)
    save.save_model(model, 'checkpoint-%d' % epoch)

    batch = 1
    for data in train_dataloader:
        if args.model == 'base':
            predicted = model(*data)
            loss = model.loss(*predicted[:2], data[0], predicted[-1])
        if args.model == 'test':
            predicted = model(data[0])
            loss = model.loss(*predicted[:2], data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save.training_log(epoch, batch, step, loss)
        if step % 100 == 0:
            save.training_attention(step, predicted[2], predicted[3])
        batch += 1
        step += 1

    continue
    with torch.no_grad():
        for data in valid_dataloader:
            predicted = model(*data)
            #labels = torch.stack(data[-1])
            loss = model.loss(*predicted[:2], data[0], predicted[-1])
            save.validation_log(epoch, step, loss)
            break
