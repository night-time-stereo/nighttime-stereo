

class DepthEstimationDataLoader(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def _get_item(self, index):
        return self.dataset[index]

    def _preprocess(self):
        raise NotImplementedError

    def __getitem__(self, index):
        args = self._get_item(index)
        return zip(args, self._preprocess(*args))
