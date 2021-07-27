import torch

class HDFSDatasetFetcher(torch.utils.data._utils.fetch._MapDatasetFetcher):

    def __init__(self, *args, num_readers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_readers = num_readers

    def fetch(self, possibly_batched_index):
        data = self.dataset.get_items(possibly_batched_index, num_readers=self.num_readers)
        return self.collate_fn(data)

class HDFSIterator(torch.utils.data.dataloader._SingleProcessDataLoaderIter):

    def __init__(self, loader, num_readers=1, **kwargs):
        super().__init__(loader, **kwargs)
        self._dataset_fetcher = HDFSDatasetFetcher(self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

class HDFSLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, num_readers=1, **kwargs):
        kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)
        self.num_readers = num_readers

    def _get_iterator(self):
        return HDFSIterator(self, num_readers=self.num_readers)

class HDFSDataset(torch.utils.data.Dataset):

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.get_items([index])

class HDFSCollate:

    def __init__(self, device=None):
        self.device = device

    def __call__(self, batch):
        if self.device is None:
            return batch

        for i, data in enumerate(batch):
            for j in range(len(data)):
                if torch.is_tensor(batch[i][j]):
                    batch[i][j] = batch[i][j].to(self.device)
                else:
                    batch[i][j] = torch.from_numpy(batch[i][j]).to(self.device)
        return batch
