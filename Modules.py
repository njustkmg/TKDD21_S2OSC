import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
from numpy import ndarray

import random
import CTAugment
from Dataset import Dataset, PoolDataset
from typing import Tuple, NoReturn


class ResNet18(nn.Module):
    def __init__(self, n_out=10, T=1):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.fc = torch.nn.Linear(in_features=512, out_features=n_out)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.T = T

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        f512 = self.resnet18(x).view(-1, 512)
        f10 = self.fc(self.relu(f512))
        f10 = self.softmax(f10)
        return f512, f10

    def load(self, filename: str) -> NoReturn:
        super(ResNet18, self).load_state_dict(torch.load(filename, map_location=torch.device("cuda")))

    def save(self, filename: str) -> NoReturn:
        torch.save(self.state_dict(), filename)


class ResNet34(nn.Module):
    def __init__(self, n_out=10, T=1):
        super(ResNet34, self).__init__()
        self.resnet34 = torchvision.models.resnet34(pretrained=True)
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-1])
        self.fc = nn.Linear(in_features=512, out_features=n_out)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.T = T

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        f512 = self.resnet34(x).view(-1, 512)
        f10 = self.fc(self.relu(f512))
        f10 = self.softmax(f10)
        return f512, f10

    def load(self, filename: str) -> NoReturn:
        super(ResNet34, self).load_state_dict(torch.load(filename, map_location=torch.device("cuda")))

    def save(self, filename: str) -> NoReturn:
        torch.save(self.state_dict(), filename)


class Centers:
    def __init__(self, rate_old: float, n_centers=10):
        self.rate_old = rate_old
        self.centers_index = set()
        self.centers = torch.zeros((n_centers, 512), dtype=torch.float32, requires_grad=False).cuda()

    def __str__(self):
        return f"centers_index = {self.centers_index}    shape = {self.centers.shape}"

    def init_calc(self, dataset_name: str, model: ResNet34, n_centers=10) -> NoReturn:
        dataset_name = Dataset(dataset_name=dataset_name, dataset_type="init")
        dataloader = DataLoader(dataset_name, batch_size=512, shuffle=True, num_workers=4)

        self.centers_index = set()
        self.centers = torch.zeros((n_centers, 512), dtype=torch.float32, requires_grad=False).cuda()

        with torch.no_grad():
            for _, batch_data, batch_labels in dataloader:
                batch_data = batch_data.cuda()
                batch_features, _ = model(batch_data)
                for k in set(batch_labels.tolist()):
                    k = int(k)
                    k_features = batch_features[batch_labels == k]
                    center = torch.mean(k_features, dim=0)
                    if k in self.centers_index:
                        self.centers[k] = self.rate_old * self.centers[k] + (1 - self.rate_old) * center
                    else:
                        self.centers_index.add(k)
                        self.centers[k] = center

    def update(self, dataset: PoolDataset, model: ResNet34) -> NoReturn:
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

        with torch.no_grad():
            for _, batch_data, batch_labels in dataloader:
                batch_data = batch_data.cuda()
                batch_features, _ = model(batch_data)
                for k in set(batch_labels.tolist()):
                    k = int(k)
                    k_features = batch_features[batch_labels == k]
                    center = torch.mean(k_features, dim=0)
                    if k in self.centers_index:
                        self.centers[k] = self.rate_old * self.centers[k] + (1 - self.rate_old) * center
                    else:
                        self.centers_index.add(k)
                        self.centers[k] = center

    def load(self, filename: str) -> NoReturn:
        centers_files = np.load(filename)
        self.centers_index = set(centers_files["centers_index"].tolist())
        centers_array = centers_files["centers_array"]
        self.centers = torch.tensor(centers_array, dtype=torch.float32, requires_grad=False).cuda()

    def save(self, filename: str) -> NoReturn:
        centers_index = np.array(list(self.centers_index), dtype=np.int32)
        centers_array = self.centers.cpu().numpy()
        np.savez(filename, centers_index=centers_index, centers_array=centers_array)


class Memory:
    def __init__(self, K: int):
        self.data = dict()
        self.K = K

    def __str__(self):
        count_list = []
        for k, k_data in self.data.items():
            count_list.append(f"C{k}:{k_data.shape[0]}")
        count_str = " ".join(count_list)
        return count_str

    def init_fill(self, dataset_name: str) -> NoReturn:
        dataset_name = Dataset(dataset_name=dataset_name, dataset_type="init")
        data = np.array(dataset_name.data, dtype=np.float32)
        labels = np.array(dataset_name.labels, dtype=np.int32)
        for k in set(labels.tolist()):
            k = int(k)
            k_data = data[labels == k]
            random_index = random.sample(list(range(k_data.shape[0])), k=k_data.shape[0])
            self.data[k] = k_data[random_index][:self.K]

    def update(self, dataset: PoolDataset) -> NoReturn:
        data = np.array(dataset.raw_data, dtype=np.float32)
        labels = np.array(dataset.labels, dtype=np.int32)
        for k in set(labels.flatten().tolist()):
            k = int(k)
            k_data = data[labels == k]
            if k in self.data.keys():
                k_data = np.r_[self.data[k], k_data]
            random_index = random.sample(list(range(k_data.shape[0])), k=k_data.shape[0])
            self.data[k] = k_data[random_index][:self.K]

    def get_data(self, num: int = -1) -> Tuple[ndarray, ndarray]:
        data = None
        labels = None
        for k, k_data in self.data.items():
            k_data = k_data[:num]
            k_labels = np.full(k_data.shape[0], k, dtype=np.int32)[:num]
            data = k_data if data is None else np.r_[data, k_data]
            labels = k_labels if labels is None else np.r_[labels, k_labels]
        return data, labels

    def load(self, filename: str) -> NoReturn:
        memory_files = np.load(filename)
        for k in memory_files.files:
            self.data[int(k)] = memory_files[k]

    def save(self, filename: str) -> NoReturn:
        memory_files = {}
        for k, k_data in self.data.items():
            memory_files[str(k)] = k_data
        np.savez(filename, **memory_files)


class Augment:
    @staticmethod
    def weak_augment(data: Tensor) -> Tensor:
        # transform_0 = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomHorizontalFlip(p=1.0),
        #     transforms.RandomRotation(degrees=10),
        # ])

        transform_1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])

        data = data.cpu()
        augment_data = torch.zeros_like(data, dtype=torch.float32).cuda()
        for i, raw_data in enumerate(data):
            # if i < 100:
            #     transform_0(raw_data).save(f"running/weak/{i}.jpg")
            augment_data[i] = transform_1(raw_data)

        return augment_data

    @staticmethod
    def strong_augment(data: Tensor) -> Tensor:
        data = data.cpu().numpy().transpose(0, 2, 3, 1)
        augment_data = np.zeros_like(data, dtype=np.float32)
        cta = CTAugment.CTAugment()

        for i, raw_data in enumerate(data):
            augment_data[i] = CTAugment.apply(raw_data, cta.policy(probe=True))
            # if i < 100:
            #     Image.fromarray((augment_data[i] * 255).astype("uint8")).save(f"running/strong/{i}.jpg")

        augment_data = torch.tensor(augment_data.transpose(0, 3, 1, 2), dtype=torch.float32).cuda()
        return augment_data
