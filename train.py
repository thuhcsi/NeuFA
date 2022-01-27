import os
import sys
import torch
import argparse
from tqdm import tqdm
from model.neufa import NeuFA_base, NeuFA_TeP, NeuFA_MeP, NeuFA_TeMP
from data.common import Collate
from save import Save

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', default=None)
parser.add_argument('--load_model', default=None)
parser.add_argument('--train_path', default=os.path.expanduser('~/LibriSpeech'))
parser.add_argument('--dev_path', default=os.path.expanduser('~/BuckeyeTrain'))
parser.add_argument('--valid_path', default=os.path.expanduser('~/BuckeyeTest'))
parser.add_argument('--model', default='temp', choices=['base', 'tep', 'mep', 'temp'])
args = parser.parse_args()

device = "cuda:%d" % args.gpu

if args.model == 'base':
    from hparams import base as hparams
    model = NeuFA_base(hparams)
elif args.model == 'tep':
    from hparams import base as hparams
    model = NeuFA_TeP(hparams)
elif args.model == 'mep':
    from hparams import base as hparams
    model = NeuFA_MeP(hparams)
elif args.model == 'temp':
    from hparams import temp as hparams
    model = NeuFA_TeMP(hparams)

if hparams.strategy != 'finetune':
    if 'LJSpeech' in args.train_path:
        from data.ljspeech import LJSpeech
        train_dataset = LJSpeech(args.train_path, reduction=hparams.reduction_rate)
    elif 'LibriSpeech' in args.train_path:
        from data.librispeech import LibriSpeech
        train_dataset = LibriSpeech(args.train_path, reduction=hparams.reduction_rate)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)

if hparams.strategy != 'pretrain':
    from data.buckeye import Buckeye, BuckeyePhoneme
    dev_dataset = Buckeye(args.dev_path, reduction=hparams.reduction_rate)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)
    valid_dataset = Buckeye(args.valid_path, reduction=hparams.reduction_rate)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)

if hparams.strategy == 'semi2':
    dev_dataset2 = BuckeyePhoneme(args.dev_path, reduction=hparams.reduction_rate)
    dev_dataloader2 = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)
    valid_dataset2 = BuckeyePhoneme(args.valid_path, reduction=hparams.reduction_rate)
    valid_dataloader2 = torch.utils.data.DataLoader(valid_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)

if args.load_model:
    #model_dict = model.state_dict()
    #state_dict = torch.load(args.load_model)
    #state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aligner.')}
    #model_dict.update(state_dict)
    #model.load_state_dict(model_dict)
    model.load_state_dict(torch.load(args.load_model, map_location='cpu'))

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if args.name is None:
    args.name = args.model
#else:
#    args.name = args.model + '_' + args.name
save = Save(args.name)
save.save_parameters(hparams)

def process(model, stage, data, step, batch):
    if isinstance(model, NeuFA_base):
        predicted = model(*data[:2])
        text_loss = model.text_loss(predicted[0], data[0])
        save.writer.add_scalar(f'{stage}/text loss', text_loss, step)
        speech_loss = model.mfcc_loss(predicted[1], data[1])
        save.writer.add_scalar(f'{stage}/speech loss', speech_loss, step)
        loss = hparams.text_loss * text_loss + hparams.speech_loss * speech_loss
        if stage == 'training':
            attention_loss = model.attention_loss(*predicted[2:4], hparams.attention_loss_alpha)
            save.writer.add_scalar(f'{stage}/attention loss', attention_loss, step)
            loss += hparams.attention_loss * attention_loss
        else:
            boundary_loss = model.boundary_loss(predicted[-1], data[2])
            save.writer.add_scalar(f'{stage}/boundary loss', boundary_loss, step)
            boundaries = model.extract_boundary(predicted[-1])
            boundary_mae = model.boundary_mae(boundaries, data[2])
            save.writer.add_scalar(f'{stage}/boundary mae', boundary_mae, step)
            loss += hparams.boundary_loss * boundary_loss
    if isinstance(model, NeuFA_TeP):
        tep_loss = model.length_loss(*predicted[4:6])
        tep_mse = model.length_loss(*predicted[4:6], normalize=False)
        save.writer.add_scalar(f'{stage}/tep loss', tep_loss, step)
        save.writer.add_scalar(f'{stage}/tep rmse', torch.sqrt(tep_mse), step)
        loss += hparams.tep_loss * tep_loss
    if isinstance(model, NeuFA_MeP):
        if isinstance(model, NeuFA_TeMP):
            mep_loss = model.length_loss(*predicted[6:8])
            mep_mse = model.length_loss(*predicted[6:8], normalize=False)
        else:
            mep_loss = model.length_loss(*predicted[4:6])
            mep_mse = model.length_loss(*predicted[4:6], normalize=False)
        save.writer.add_scalar(f'{stage}/mep loss', mep_loss, step)
        save.writer.add_scalar(f'{stage}/mep rmse', torch.sqrt(mep_mse), step)
        loss += hparams.mep_loss * mep_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    save.save_log(stage, epoch, batch, step, loss)
    if step % 100 == 0:
        save.save_attention(stage, step, predicted[2][0], predicted[3][0])
        if stage != 'training':
            save.save_boundary(stage, step, boundaries[0], data[2][0], predicted[2][0].shape)
    if step % 20000 == 0:
        save.save_model(model, f'{step // 1000}k')

step = 1
for epoch in range(hparams.max_epochs):
    save.logger.info('Epoch %d', epoch)

    batch = 1

    if hparams.strategy == 'pretrain':
        for data in train_dataloader:
            process(model, 'training', data, step, batch)
            step += 1
            batch += 1
        continue

    for data in dev_dataloader:
        if hparams.strategy == 'finetune':
            process(model, 'dev', data, step, batch)
        if hparams.strategy == 'semi':
            training_data = next(iter(train_dataloader))
            process(model, 'training', training_data, step, batch)
            process(model, 'dev', data, step, batch)
        if hparams.strategy == 'semi2':
            training_data = next(iter(train_dataloader))
            dev_data2 = next(iter(dev_dataloader2))
            process(model, 'training', training_data, step, batch)
            process(model, 'dev', data, step, batch)
            process(model, 'dev2', dev_data2, step, batch)
        batch += 1
        step += 1

    with torch.no_grad():
        predicted = []
        all_data = []
        for data in tqdm(valid_dataloader):
            all_data.append(data)
            predicted.append(model(*data[:2]))

        data = [i for i in zip(*all_data)]
        predicted = [i for i in zip(*predicted)]

        for i in range(len(data)):
            data[i] = [k for j in data[i] for k in j]

        for i in range(len(predicted)):
            predicted[i] = [k for j in predicted[i] for k in j]

        if isinstance(model, NeuFA_base):
            text_loss = model.text_loss(predicted[0], data[0])
            save.writer.add_scalar('test/text loss', text_loss, epoch)
            speech_loss = model.mfcc_loss(predicted[1], data[1])
            save.writer.add_scalar('test/speech loss', speech_loss, epoch)
            attention_loss = model.attention_loss(*predicted[2:4], 1)
            save.writer.add_scalar('test/attention loss', attention_loss, epoch)
            loss = hparams.text_loss * text_loss + hparams.speech_loss * speech_loss + hparams.attention_loss * attention_loss
            boundary_loss = model.boundary_loss(predicted[-1], data[2])
            save.writer.add_scalar('test/boundary loss', boundary_loss, epoch)
            boundaries = model.extract_boundary(predicted[-1])
            boundary_mae = model.boundary_mae(boundaries, data[2])
            save.writer.add_scalar(f'test/boundary mae', boundary_mae, epoch)
            loss += hparams.boundary_loss * boundary_loss
        if isinstance(model, NeuFA_TeP):
            tep_loss = model.length_loss(*predicted[4:6])
            tep_mse = model.length_loss(*predicted[4:6], normalize=False)
            save.writer.add_scalar('test/tep loss', tep_loss, epoch)
            save.writer.add_scalar('test/tep rmse', torch.sqrt(tep_mse), epoch)
            loss += hparams.tep_loss * tep_loss
        if isinstance(model, NeuFA_MeP):
            if isinstance(model, NeuFA_TeMP):
                mep_loss = model.length_loss(*predicted[6:8])
                mep_mse = model.length_loss(*predicted[6:8], normalize=False)
            else:
                mep_loss = model.length_loss(*predicted[4:6])
                mep_mse = model.length_loss(*predicted[4:6], normalize=False)
            save.writer.add_scalar('test/mep loss', mep_loss, epoch)
            save.writer.add_scalar('test/mep rmse', torch.sqrt(mep_mse), epoch)
            loss += hparams.mep_loss * mep_loss
        save.save_log('test', epoch, batch, epoch, loss)
        save.save_attention('test', epoch, predicted[2][0], predicted[3][0])
        save.save_boundary('test', epoch, boundaries[0], data[2][0], predicted[2][0].shape)
