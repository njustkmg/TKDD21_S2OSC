import torch
from torch import nn
from torch.utils.data import DataLoader

import os
from sklearn import metrics
from logging import Logger

from Dataset import Dataset
from Modules import ResNet34, Memory, Centers


def d(dataset_name: str) -> str:
    if dataset_name in ["m", "MNIST"]:
        dataset_name = "MNIST"
    elif dataset_name in ["fm", "FASHIONMNIST"]:
        dataset_name = "FASHIONMNIST"
    elif dataset_name in ["c10", "CIFAR10"]:
        dataset_name = "CIFAR10"
    elif dataset_name in ["cinic", "CINIC"]:
        dataset_name = "CINIC"
    elif dataset_name in ["svhn", "SVHN"]:
        dataset_name = "SVHN"
    else:
        raise KeyError('"dataset" must be in ["m", "MNIST", "fm", "FASHIONMNIST", "c10",'' "CIFAR10",' +
                       ' "cinic", "CINIC", "svhn", "SVHN"].')
    return dataset_name


def load_or_init_train(logger: Logger, dataset_name: str, train: bool, n_out: int,
                       model: ResNet34, memory: Memory, centers: Centers, running_path: str,
                       epochs: int, batch_size: int, lr: float, sgd_beta: float, sgd_decay: float):
    if not os.path.exists(running_path):
        os.makedirs(running_path)

    model_file = f"{running_path}/{dataset_name}_init.pth"
    centers_file = f"{running_path}/{dataset_name}_init.centers.npz"
    memory_file = f"{running_path}/{dataset_name}_init.memory.npz"

    if train:
        model = init_train(logger=logger, dataset_name=dataset_name, model=model, epochs=epochs,
                           batch_size=batch_size, lr=lr, sgd_beta=sgd_beta, sgd_decay=sgd_decay)
        model.save(model_file)
        logger.info(f'Saved Model_F to "{model_file}".')

        logger.debug("Filling Memory ...")
        memory.init_fill(dataset_name=dataset_name)
        logger.debug(f"Finished Filling Memory: {memory.__str__()}.")
        memory.save(memory_file)
        logger.info(f'Saved Memory to "{memory_file}".')

        logger.debug("Calculating Centers ...")
        centers.init_calc(dataset_name=dataset_name, model=model, n_centers=n_out)
        logger.debug(f"Finished Calculating Centers: {centers.__str__()}.")
        centers.save(centers_file)
        logger.info(f'Saved Centers to "{centers_file}".')
    else:
        model.load(model_file)
        logger.info(f'Loaded Model_F from "{model_file}".')
        memory.load(memory_file)
        logger.info(f'Loaded Memory from "{memory_file}".')
        centers.load(centers_file)
        logger.info(f'Loaded Centers from "{centers_file}".')


def init_train(logger: Logger, dataset_name: str, model: ResNet34, epochs: int,
               batch_size: int, lr: float, sgd_beta: float, sgd_decay: float) -> ResNet34:
    logger.debug("--------    Start Initial Training    --------")

    dataset = Dataset(dataset_name=dataset_name, dataset_type="init")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=sgd_beta, weight_decay=sgd_decay)

    for epoch in range(1, epochs + 1):
        y_pred = []
        y_true = []
        for iteration, (_, batch_data, batch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda().long()
            _, batch_out = model(batch_data)
            loss = criterion(batch_out, batch_labels)
            loss.backward()
            optimizer.step()
            y_pred.extend(batch_out.max(dim=1)[1].tolist())
            y_true.extend(batch_labels.tolist())

            if iteration % 100 == 0:
                logger.debug(f"    loss = {loss:8.5f}")

        acc = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
        m = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)
        if epoch == epochs:
            logger.info(f"Epoch {epoch:3d} / {epochs:<3d}    Accuracy = {acc * 100:5.2f}% ")
            logger.info(f"\n{m}")
        else:
            logger.debug(f"Epoch {epoch:3d} / {epochs:<3d}    Accuracy = {acc * 100:5.2f}% ")
            logger.debug(f"\n{m}")

    return model
