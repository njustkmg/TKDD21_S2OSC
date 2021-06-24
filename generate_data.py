import torchvision
import torchvision.transforms as transforms

import numpy as np
from numpy import ndarray

import os
import random
import logging
import argparse

from logging import Logger
from typing import Set, NoReturn


def set_logger(filename: str) -> Logger:
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    log.addHandler(console_handler)
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)
    return log


class Dataset:
    def __init__(self, dataset_name, dataset_path: str):
        self.dataset_name = dataset_name
        transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert(mode="RGB") if image.mode != "RGB" else image),
            transforms.ToTensor(),
        ])

        # load data

        if dataset_name == "MNIST":
            train_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, transform=transform)
        elif dataset_name == "FASHIONMNIST":
            train_dataset = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, transform=transform)
        elif dataset_name == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, transform=transform)
        elif dataset_name == "SVHN":
            train_dataset = torchvision.datasets.SVHN(root=dataset_path, split="train", transform=transform)
            test_dataset = torchvision.datasets.SVHN(root=dataset_path, split="test", transform=transform)
        elif dataset_name == "CINIC":
            train_dataset_path = os.path.join(dataset_path, "train")
            test_dataset_path = os.path.join(dataset_path, "test")
            train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)
            test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform)
        else:
            raise KeyError('"dataset_name" must be in ["CIFAR10", "MNIST", "FASHIONMNIST", "SVHN", "CINIC"].')

        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.data_shape = train_dataset[0][0].shape

        # fill data

        self.train_data = np.zeros(len(train_dataset), dtype=[
            ("data", np.float32, self.data_shape),
            ("label", np.int32)
        ])
        self.test_data = np.zeros(len(test_dataset), dtype=[
            ("data", np.float32, self.data_shape),
            ("label", np.int32)
        ])

        for i, (data, label) in enumerate(train_dataset):
            data = np.array(data, dtype=np.float32)
            assert data.shape == self.data_shape
            self.train_data[i]["data"] = data
            self.train_data[i]["label"] = label
        for i, (data, label) in enumerate(test_dataset):
            data = np.array(data, dtype=np.float32)
            assert data.shape == self.data_shape
            self.test_data[i]["data"] = data
            self.test_data[i]["label"] = label

        self.data = np.r_[self.train_data, self.test_data]

        # classify data

        self.class_train_data = dict()
        for k, _ in enumerate(classes):
            self.class_train_data[k] = self.train_data[self.train_data["label"] == k]

        self.class_test_data = dict()
        for k, _ in enumerate(classes):
            self.class_test_data[k] = self.test_data[self.test_data["label"] == k]

        self.class_data = dict()
        for i in classes:
            self.class_data[i] = np.r_[self.class_train_data[i], self.class_test_data[i]]

    def __str__(self):
        s = f"{self.dataset_name}    {self.data_shape}\n"

        data_list = [
            ("all", self.data, self.class_data),
            ("train", self.train_data, self.class_train_data),
            ("test", self.test_data, self.class_test_data)
        ]

        for name, data, class_data in data_list:
            s += f"    {name:<5s} {data.shape[0]:<6d} "
            s += " ".join([f"C{k}:{data_array.shape[0]:<5d}" for k, data_array in class_data.items()])
            s += "\n"

        s = s.rstrip("\n")
        return s


def show_data(data: ndarray) -> str:
    s = f"{data.shape[0]:<5d} "
    for k in set(data["label"].tolist()):
        n_samples = np.where(data['label'] == k)[0].shape[0]
        if n_samples > 0:
            s += f"C{k}:{n_samples:<4d} "
    return s


def generate_data(dataset_name: str, original_dataset_path: str, output_dataset_path: str, log: Logger,
                  known_classes: Set[int], novel_classes: Set[int],
                  n_init: int, n_known: int, n_novel: int, **kwargs) -> NoReturn:
    dataset = Dataset(dataset_name, original_dataset_path)
    log.info(str(dataset))

    # create target folder
    if not os.path.exists(output_dataset_path):
        os.makedirs(output_dataset_path)

    # generate init data
    data = None
    for k, data_array in dataset.class_data.items():
        if k in known_classes:
            index_list = random.sample(list(range(data_array.shape[0])), k=n_init)
            k_data = data_array[index_list]
            data = np.r_[data, k_data] if data is not None else k_data
            index_list = list(set(range(data_array.shape[0])).difference(index_list))
            dataset.class_data[k] = dataset.class_data[k][index_list]

    # save init file
    log.info("init   " + show_data(data))
    filename = os.path.join(output_dataset_path, f"{dataset_name.lower()}_init.npy")
    np.save(filename, data)
    log.info(f'successfully saved "{filename}".')

    # generate stream data
    data = None
    for k, data_array in dataset.class_data.items():
        if k in known_classes:
            index_list = random.sample(list(range(data_array.shape[0])), k=n_known)
        elif k in novel_classes:
            index_list = random.sample(list(range(data_array.shape[0])), k=n_novel)
        else:
            continue
        k_data = data_array[index_list]
        data = np.r_[data, k_data] if data is not None else k_data
        index_list = list(set(range(data_array.shape[0])).difference(index_list))
        dataset.class_data[k] = dataset.class_data[k][index_list]

    # save stream file
    log.info("stream " + show_data(data))
    filename = os.path.join(output_dataset_path, f"{dataset_name.lower()}_stream.npy")
    np.save(filename, data)
    log.info(f'successfully saved "{filename}".\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation for S2OSC")
    parser.add_argument("--dataset", type=str, default="c10", choices=["m", "fm", "c10", "cinic", "svhn"])
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    if args.dataset == "m":
        dataset_name_ = "MNIST"
    elif args.dataset == "fm":
        dataset_name_ = "FASHIONMNIST"
    elif args.dataset == "c10":
        dataset_name_ = "CIFAR10"
    elif args.dataset == "cinic":
        dataset_name_ = "CINIC"
    elif args.dataset == "svhn":
        dataset_name_ = "SVHN"
    else:
        raise KeyError('dataset must be in ["m", "fm", "c10", "cinic", "svhn"].')

    args = {
        "dataset_name": dataset_name_,
        "original_dataset_path": args.path,
        "output_dataset_path": f"./DataSets/{dataset_name_}",
        "known_classes": {0, 1, 2, 3, 4},
        "novel_classes": {5},
        "n_init": 4000,
        "n_known": 2000,
        "n_novel": 6000,
    }

    if not os.path.exists(args["output_dataset_path"]):
        os.makedirs(args["output_dataset_path"])

    log_filename = os.path.join(args["output_dataset_path"], "data_info.txt")
    logger = set_logger(filename=log_filename)
    args["log"] = logger

    for key, value in args.items():
        logger.info(f"{key}: {value}")
    logger.info("")

    generate_data(**args)
