import torch

class Collate:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, batch):
        length = len(batch[0])
        output = [[] for i in range(length)]

        for data in batch:
            for i, j in enumerate(data):
                if not torch.is_tensor(j):
                    j = torch.from_numpy(j)
                output[i].append(j if self.device is None else j.to(self.device))

        return tuple(output)
