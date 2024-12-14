from sklearn.datasets import load_svmlight_file
import torch
from torch.utils.data import Dataset


class LIBSVM(Dataset):
    def __init__(self, path):
        super().__init__()
        # Load the dataset
        data, targets = load_svmlight_file(path, n_features=123)

        # Convert targets for values in {0, 1}
        targets = (targets + 1) / 2

        # Convert to PyTorch tensors
        data = torch.tensor(data.toarray(), dtype=torch.float32)
        targets = torch.tensor((targets + 1) // 2, dtype=torch.long)

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index].item()
