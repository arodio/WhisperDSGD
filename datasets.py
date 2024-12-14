import os
import pickle
import string

from data.a9a.libsvm import LIBSVM

import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__

    """

    def __init__(self, path):
        """
        :param path: path to .pkl file

        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx


class SubA9A(Dataset):
    """
    Constructs a subset of the a9a dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices : list of indices
    data : torch.Tensor
    targets : torch.Tensor

    Methods
    -------
    __init__(path, a9a_data=None, a9a_targets=None)
    __len__()
    __getitem__(index)
    """

    def __init__(self, path, a9a_data=None, a9a_targets=None):
        """
        Initialize SubA9A dataset.

        Parameters
        ----------
        path : str
            Path to the pickle file storing indices.
        a9a_data : torch.Tensor, optional
            Tensor containing the full a9a dataset features.
        a9a_targets : torch.Tensor, optional
            Tensor containing the full a9a dataset labels.
        """

        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if a9a_data is None or a9a_targets is None:
            self.data, self.targets = get_a9a()
        else:
            self.data, self.targets = a9a_data, a9a_targets

        self.data = a9a_data[self.indices]
        self.targets = a9a_targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index


class SubMNIST(Dataset):
    """
    Constructs a subset of MNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__

    """

    def __init__(
            self,
            path,
            mnist_data=None,
            mnist_targets=None,
            transform=None
    ):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param mnist_data: MNIST dataset inputs stored as torch.tensor
        :param mnist_targets: MNIST dataset labels stored as torch.tensor
        :param transform:

        """

        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if mnist_data is None or mnist_targets is None:
            self.data, self.targets = get_mnist()
        else:
            self.data, self.targets = mnist_data, mnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)
            img = img.reshape(-1)  # flatten the image, used in linear model

        target = target

        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__

    """

    def __init__(
            self,
            path,
            cifar10_data=None,
            cifar10_targets=None,
            transform=None
    ):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """

        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__

    """
    def __init__(self, path, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:

        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__

    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.uint8(img.numpy() * 255)
        img = Image.fromarray(img, mode='L').resize((32, 32)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input

        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences

        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


def get_a9a():
    """
    gets full (both train and test) a9a dataset inputs and labels;
    the dataset should be first downloaded (see data/a9a)

    :return:
        a9a_data, a9a_targets

    """
    a9a_path = os.path.join("data", "a9a", "raw_data")
    assert os.path.isdir(a9a_path), "Download the a9a dataset!!"

    # Load train and test datasets
    a9a_train = LIBSVM(os.path.join(a9a_path, "a9a.txt"))
    a9a_test = LIBSVM(os.path.join(a9a_path, "a9a_t.txt"))

    # Combine data and targets
    a9a_data = \
        torch.cat([
            a9a_train.data.clone().detach(),
            a9a_test.data.clone().detach()
        ])
    a9a_targets = \
        torch.cat([
            a9a_train.targets.clone().detach(),
            a9a_test.targets.clone().detach()
        ])

    return a9a_data, a9a_targets


def get_mnist():
    """
    gets full (both train and test) MNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/mnist/README.md)

    :return:
        mnist_data, mnist_targets

    """
    mnist_path = os.path.join("data", "mnist", "raw_data")
    assert os.path.isdir(mnist_path), "Download MNIST dataset!!"

    mnist_train = \
        MNIST(
            root=mnist_path,
            train=True,
            download=False
        )

    mnist_test = \
        MNIST(
            root=mnist_path,
            train=False,
            download=False
        )

    mnist_data = \
        torch.cat([
            mnist_train.data.clone().detach(),
            mnist_test.data.clone().detach()
        ])

    mnist_targets = \
        torch.cat([
            mnist_train.targets.clone().detach(),
            mnist_test.targets.clone().detach()
        ])

    return mnist_data, mnist_targets


def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)

    :return:
        cifar10_data, cifar10_targets

    """
    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)

    :return:
        cifar100_data, cifar100_targets

    """
    cifar100_path = os.path.join("data", "cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download cifar100 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets
