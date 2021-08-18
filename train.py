import os
import torch
import argparse
from dataset import LibriSpeech, LibriSpeechText
from hdfs import HDFSLoader, HDFSCollate
from save import Save

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', default=None)
parser.add_argument('--restore_path', default=None)
parser.add_argument('--model', required=True, choices=['base', 'test', 'test2', 'test3', 'test4'])
args = parser.parse_args()

device = "cuda:%d" % args.gpu

if args.model == 'base':
    from model.neufa import NeuFA_base as Model
    from hparams import base as hparams
    #train_dataset = LibriSpeech('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    train_dataset = LibriSpeech(os.path.expanduser('~/packed/LibriSpeech'))
elif args.model == 'test':
    from model.test import BidirectionalAttentionTest as Model
    from hparams import test as hparams
    train_dataset = LibriSpeechText('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    #train_dataset = LibriSpeechText(os.path.expanduser('~/datasets/LibriSpeech/packed/LibriSpeech'))
elif args.model == 'test2':
    from model.test import BidirectionalAttentionTest2 as Model
    from hparams import test2 as hparams
    #train_dataset = LibriSpeechText('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    train_dataset = LibriSpeechText(os.path.expanduser('~/packed/LibriSpeech'))
elif args.model == 'test3':
    from model.test import SelfAttentionTest as Model
    from hparams import test as hparams
    train_dataset = LibriSpeechText('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    #train_dataset = LibriSpeechText(os.path.expanduser('~/datasets/LibriSpeech/packed/LibriSpeech'))
elif args.model == 'test4':
    from model.test import SelfAttentionTest2 as Model
    from hparams import test as hparams
    train_dataset = LibriSpeechText('hdfs://haruna/home/byte_speech_sv/user/lijingbei/LibriSpeech/packed/LibriSpeech')
    #train_dataset = LibriSpeechText(os.path.expanduser('~/datasets/LibriSpeech/packed/LibriSpeech'))

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
else:
    args.name = args.model + '_' + args.name
save = Save(args.name)

step = 1
for epoch in range(hparams.max_epochs):
    save.logger.info('Epoch %d', epoch)
    save.save_model(model, 'checkpoint-%d' % epoch)

    batch = 1
    for data in train_dataloader:
        if args.model == 'base':
            predicted = model(*data)
            text_loss = hparams.text_loss * model.text_loss(predicted[0], data[0])
            save.writer.add_scalar('text loss', text_loss, step)
            speech_loss = hparams.speech_loss * model.mfcc_loss(predicted[1], predicted[-1])
            save.writer.add_scalar('speech loss', speech_loss, step)
            attention_loss = hparams.attention_loss * model.attention_loss(*predicted[2:4], 1)
            save.writer.add_scalar('attention loss', attention_loss, step)
            loss = text_loss + speech_loss + attention_loss
        elif args.model.startswith('test'):
            predicted = model(data[0])
            attention_loss = hparams.attention_loss * model.attention_loss(*predicted[2:4])
            save.writer.add_scalar('attention loss', attention_loss, step)
            loss = model.loss(*predicted[:2], data[0])
            save.writer.add_scalar('cross entropy loss', loss, step)
            loss += attention_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save.training_log(epoch, batch, step, loss)
        if step % 100 == 0:
            save.training_attention(step, predicted[2][0], predicted[3][0])
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
