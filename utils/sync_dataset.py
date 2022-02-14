from torch.utils.data import Dataset

class syncDataset(Dataset):
    def __init__(self, *kwargs):
        self.kwargs = kwargs
        assert min([len(dataset) for dataset in self.kwargs]) == max([len(dataset) for dataset in self.kwargs])
    def __getitem__(self, index):
        return [dataset[index] for dataset in self.kwargs]
    def __len__(self):
        return len(self.kwargs[0])