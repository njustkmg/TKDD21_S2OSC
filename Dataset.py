import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch import Tensor

import numpy as np

import os
from typing import Tuple, Optional

# "DATASETS_PATH" is the parent path of all datasets' folders.
DATASETS_PATH = "./DataSets"


# All data files are stored in numpy format using "np.load()" and "np.save()".
# The tree is shown as below:
#
# ROOT
# │
# ├────main.py
# ├────Dataset.py (This File)
# ├────......
# ├────......
# │
# └────DataSets
#      ├────CIFAR10
#      │    ├────cifar10_init.npy
#      │    ├────cifar10_stream.npy
#      │    └────cifar10_test.npy
#      ├────CINIC
#      │    ├────cinic_init.npy
#      │    ├────cinic_stream.npy
#      │    └────cinic_test.npy
#      ├────SVHN
#      │    ├────svhn_init.npy
#      │    ├────svhn_stream.npy
#      │    └────svhn_test.npy
#      ├────MNIST
#      │    ├────mnist_init.npy
#      │    ├────mnist_stream.npy
#      │    └────mnist_test.npy
#      └────FASHIONMNIST
#           ├────fashionmnist_init.npy
#           ├────fashionmnist_stream.npy
#           └────fashionmnist_test.npy


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, dataset_type: str):
        # check validation
        if dataset_name in ["m", "MNIST"]:
            dataset_name = "MNIST"
        elif dataset_name in ["fm", "FASHIONMNIST"]:
            dataset_name = "FASHIONMNIST"
        elif dataset_name in ["c10", "CIFAR10"]:
            dataset_name = "CIFAR10"
        elif dataset_name in ["c100", "CIFAR100"]:
            dataset_name = "CIFAR100"
        elif dataset_name in ["cinic", "CINIC"]:
            dataset_name = "CINIC"
        elif dataset_name in ["svhn", "SVHN"]:
            dataset_name = "SVHN"
        else:
            raise KeyError('"dataset" must be in ["CIFAR10", "CINIC", "SVHN", "MNIST", "FASHIONMNIST"].')
        if dataset_type not in ["init", "stream", "test"]:
            raise KeyError('"split" must be in ["init", "stream", "test"].')

        # set transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        # load data
        filename = os.path.join(DATASETS_PATH, dataset_name.upper(), f"{dataset_name.lower()}_{dataset_type}.npy")
        dataset = np.load(filename)
        self.data = torch.tensor(dataset["data"], dtype=torch.float32)
        self.labels = torch.tensor(dataset["label"], dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        raw_data = self.data[idx]
        data_224 = self.transform(raw_data)
        label = self.labels[idx]
        return raw_data, data_224, label


class PoolDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data: Optional[Tensor], data: Optional[Tensor], labels: Optional[Tensor]):
        self.raw_data = raw_data if raw_data is not None else raw_data
        self.data = data if data is not None else data
        self.labels = labels.long() if labels is not None else labels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        if self.raw_data is not None:
            return self.raw_data.size(0)
        elif self.data is not None:
            return self.data.size(0)
        else:
            return self.labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        raw_data = self.raw_data[idx] if self.raw_data is not None else torch.tensor(0.)
        data_224 = self.data[idx] if self.data is not None else self.transform(raw_data)
        landmark = self.labels[idx] if self.labels is not None else torch.tensor(0.)
        return raw_data, data_224, landmark
