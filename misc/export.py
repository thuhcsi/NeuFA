import sys
import torch
from model.neufa import NeuFA_TeMP
from hparams import temp as hparams

model = NeuFA_TeMP(hparams)
model.load_state_dict(torch.load(sys.argv[1]))
torch.save(model, sys.argv[2])
