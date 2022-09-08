

class DepthEstimationDataLoader(object):
    def __init__(self) -> None:
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
