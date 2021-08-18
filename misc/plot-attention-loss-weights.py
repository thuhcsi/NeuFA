import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

w = torch.zeros(100, 300)
a = torch.linspace(1e-6, 1, w.shape[0]).to(w.device).repeat(w.shape[1], 1)
b = torch.linspace(1e-6, 1, w.shape[1]).to(w.device).repeat(w.shape[0], 1).T
r1 = torch.maximum((a / b), (b / a))
r2 = torch.maximum(a.flip(1) / b.flip(0), b.flip(0)/ a.flip(1))
r = torch.maximum(r1, r2) - 1
r = torch.tanh(0.5 * r)
print(r)

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(r, aspect='auto', origin='lower', interpolation='none')
fig.colorbar(im, ax=ax)
plt.tight_layout()
fig.canvas.draw()
plt.savefig('a.png')
