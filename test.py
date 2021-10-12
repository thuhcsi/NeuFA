import os
import sys
import torch
import argparse
from tqdm import tqdm
from model.neufa import NeuFA_base, NeuFA_TeP, NeuFA_MeP, NeuFA_TeMP
from data.common import Collate
from save import Save
from g2p.en_us import G2P

g2p= G2P()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--load_model', default=None)
parser.add_argument('--test_path', default=os.path.expanduser('~/BuckeyeTest1'))
parser.add_argument('--model', default='temp', choices=['base', 'tep', 'mep', 'temp'])
args = parser.parse_args()

device = "cuda:%d" % args.gpu
#device = "cpu:0"

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

from data.buckeye import Buckeye, BuckeyePhoneme
test_dataset = Buckeye(args.test_path, reduction=hparams.reduction_rate)
test_dataset = BuckeyePhoneme(args.test_path, reduction=hparams.reduction_rate)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False, collate_fn=Collate(device))


model.load_state_dict(torch.load(args.load_model))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

with torch.no_grad():
    predicted = []
    all_data = []
    for data in tqdm(test_dataloader):
        all_data.append(data)
        predicted.append(model(*data[:2]))

    data = [i for i in zip(*all_data)]
    predicted = [i for i in zip(*predicted)]

    for i in range(len(data)):
        data[i] = [k for j in data[i] for k in j]

    for i in range(len(predicted)):
        predicted[i] = [k for j in predicted[i] for k in j]

    predicted[-1] = model.extract_boundary(predicted[-1])

    print(model.boundary_mae(predicted[-1], data[2]))
    sys.exit()

    p = [predicted[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 1500]
    q = [data[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 1500]
    print(1500, len(p), model.boundary_mae(p, q))
    p = [predicted[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 1000]
    q = [data[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 1000]
    print(1000, len(p), model.boundary_mae(p, q))
    p = [predicted[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 500]
    q = [data[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 500]
    print(500, len(p), model.boundary_mae(p, q))
    p = [predicted[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 250]
    q = [data[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 250]
    print(250, len(p), model.boundary_mae(p, q))
    p = [predicted[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 125]
    q = [data[-1][i] for i in range(len(data[0])) if data[1][i].shape[0] < 125]
    print(125, len(p), model.boundary_mae(p, q))

    count = [0 for i in range(1001)]
    for i in tqdm(range(len(data[0]))):
        file = test_dataset.wavs[i]
        texts, boundaries, p_boundaries = data[0][i], data[2][i], predicted[-1][i]
        texts = [g2p.id2symbol[j-1] for j in texts]
        output=[]

        for j in range(len(texts)):
            output.append([])
            output[-1].append(texts[j])
            for k in range(2):
                if boundaries[j][k] == -1:
                    output[-1].append('-')
                else:
                    output[-1].append('%.4f' % boundaries[j][k])
            output[-1].append('%.4f' % p_boundaries[j][0])
            output[-1].append('%.4f' % p_boundaries[j][1])
            for k in range(2):
                if boundaries[j][k] == -1:
                    output[-1].append('-')
                else:
                    output[-1].append('%.4f' % (p_boundaries[j][k] - boundaries[j][k]))
                    try:
                        count[int(abs(p_boundaries[j][k] - boundaries[j][k]) * 1000)] += 1
                    except:
                        count[-1] += 1
            output[-1] = '\t'.join(output[-1])

        with open(file.parent / f'{file.stem}.output.txt', 'w') as f:
            f.write('\n'.join(output))

    print(10,   sum(count[:10]), sum(count[:10])/sum(count))
    print(25,   sum(count[10:25]), sum(count[:25])/sum(count))
    print(50,   sum(count[25:50]), sum(count[:50])/sum(count))
    print(100,  sum(count[50:100]), sum(count[:100])/sum(count))
    print(200,  sum(count[100:200]), sum(count[:200])/sum(count))
    print(500,  sum(count[200:500]), sum(count[:500])/sum(count))
    print(1000, sum(count[500:1000]), sum(count[:1000])/sum(count))
