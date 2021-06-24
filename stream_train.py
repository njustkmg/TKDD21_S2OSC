import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

import numpy as np
from numpy import ndarray

import os
import warnings
from sklearn import metrics
from logging import Logger
from typing import NoReturn, Tuple

from Dataset import Dataset, PoolDataset
from Modules import ResNet18, ResNet34, Centers, Memory, Augment

warnings.filterwarnings("ignore")


class Stream:
    def __init__(self, logger: Logger, log_folder: str, dataset_name: str,
                 model_f: ResNet34, centers: Centers, memory: Memory, T: float, K: int, n_out: int,
                 epochs: int, batch_size: int, lr: float, sgd_beta: float, sgd_decay: float,
                 lamda: float, alpha: float, lamda_u: float, tao: float):
        self.logger = logger
        self.log_folder = log_folder
        self.dataset_name = dataset_name
        self.model_f = model_f
        self.centers = centers
        self.memory = memory

        self.T = T
        self.K = K
        self.n_out = n_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.sgd_beta = sgd_beta
        self.sgd_decay = sgd_decay

        self.lamda = lamda
        self.alpha = alpha
        self.lamda_u = lamda_u
        self.tao = tao

        self.pool_size = 6000
        self.n_repeat = 200
        self._pool_raw_data = None
        self._pool_data = None
        self._pool_true_labels = None

        self._max_label = None
        self._known_labels_set = None
        self._novel_labels_set = None
        self._novel_pseudo_label = None

    def train(self) -> NoReturn:
        dataset = Dataset(dataset_name=self.dataset_name, dataset_type="stream")
        dataloader = DataLoader(dataset, batch_size=self.pool_size, shuffle=True, num_workers=4)

        model_g = ResNet18(n_out=self.n_out, T=self.T).cuda()
        for iteration, (pool_raw_data, pool_data, pool_true_labels) in enumerate(dataloader):
            self.logger.info(f"--------    Window {iteration}    --------")
            self._show_data_info(pool_true_labels)
            self._pool_raw_data = pool_raw_data
            self._pool_data = pool_data
            self._pool_true_labels = pool_true_labels

            # ---- get labels set ----
            self._max_label = sorted(set(pool_true_labels.tolist()))[-1]
            self._known_labels_set = self.centers.centers_index
            self._novel_labels_set = set(self._pool_true_labels.tolist()).difference(self._known_labels_set)
            self.logger.debug(f"Max Label = {self._max_label}")
            self.logger.debug(f"Known Labels Set = {self._known_labels_set}")
            self.logger.debug(f"Novel Labels Set = {self._novel_labels_set}")

            # ---- calculate weights and split data ----
            weight_array = self._calc_weights(pool_data, pool_true_labels)
            u_raw_data, u_data, u_true_labels, out_raw_data, out_data, out_true_labels = \
                self._split_data(pool_raw_data, pool_data, pool_true_labels, weight_array)

            # TODO
            # out_raw_data = pool_raw_data[pool_true_labels == 5][:self.K]
            # out_labels = pool_true_labels[pool_true_labels == 5][:self.K].int()

            # ---- get pseudo label ----
            self._novel_pseudo_label = self._get_pseudo_label()
            out_labels = torch.full_like(out_true_labels, self._novel_pseudo_label, dtype=torch.int32)
            num_true_novel_labels = 0
            for novel_label in self._novel_labels_set:
                num_true_novel_labels += torch.sum(out_true_labels == novel_label).item()
            percent_true_novel_labels = 100.0 * num_true_novel_labels / out_true_labels.size(0)
            self.logger.info(f"Novel Pseudo Label = {self._novel_pseudo_label}")
            self.logger.info(f"Percent True Novel Labels = {percent_true_novel_labels:5.2f}%")
            self.logger.debug(f"Selected Novel labels = {out_true_labels}")

            # ---- merge data ----
            in_data, in_labels = self.memory.get_data(num=self.K)
            in_data = torch.tensor(in_data, dtype=torch.float32)
            in_labels = torch.tensor(in_labels, dtype=torch.int32)
            x_data = torch.cat([in_data, out_raw_data])
            x_labels = torch.cat([in_labels, out_labels])

            # ---- train model g ----
            self._predict(model_g)
            self._train_model_g(model_g, x_data, x_labels, u_data)

            break

    def _show_data_info(self, batch_labels: ndarray) -> NoReturn:
        labels, counts = np.unique(batch_labels, return_counts=True)
        data_info = [f"C{k}:{c}" for k, c in zip(labels, counts)]
        data_info = "    ".join(data_info)
        self.logger.info(data_info)

    def _calc_weights(self, data: Tensor, _labels: Tensor) -> ndarray:
        dataset = PoolDataset(None, data, _labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        weights_array = np.zeros(0, dtype=np.float32)
        centers = self.centers.centers[list(self.centers.centers_index)].cpu()

        with torch.no_grad():
            for _, batch_data, _label in dataloader:
                batch_features = self.model_f(batch_data.cuda())[0].cpu()
                batch_u = torch.mean(- batch_features * torch.log(batch_features), dim=1)
                batch_d = torch.zeros(batch_data.shape[0], dtype=torch.float32)
                for i, feature in enumerate(batch_features):
                    batch_d[i] = torch.min(torch.norm(feature - centers, dim=1))
                batch_w = batch_u + self.lamda * batch_d
                weights_array = np.r_[weights_array, batch_w.numpy()]

        mid_weight = np.mean(weights_array)
        weights_array = (weights_array - mid_weight) ** 2
        return weights_array

    def _split_data(self, raw_data: Tensor, data: Tensor, labels: Tensor, weights: ndarray) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        index_weight_array = np.c_[np.arange(0, weights.shape[0]), weights]
        index_weight_array = index_weight_array[np.argsort(index_weight_array[:, 1])]
        low_idx = index_weight_array[:-self.K, 0]
        high_idx = index_weight_array[-self.K:, 0]
        low_raw_data, low_data, low_labels = raw_data[low_idx], data[low_idx], labels[low_idx]
        high_raw_data, high_data, high_labels = raw_data[high_idx], data[high_idx], labels[high_idx]
        return low_raw_data, low_data, low_labels, high_raw_data, high_data, high_labels

    def _get_pseudo_label(self) -> int:
        for k in range(10):
            if k not in self.centers.centers_index:
                return k

    def _train_model_g(self, model_g: ResNet18, dataset_x_data: Tensor, dataset_x_labels: Tensor,
                       dataset_u_data: Tensor) -> ResNet18:
        self.logger.debug("----    train Model_G    ----")

        dataset_x_data_novel = dataset_x_data[dataset_x_labels == self._novel_pseudo_label][:self.n_repeat]
        dataset_x_labels_novel = dataset_x_labels[dataset_x_labels == self._novel_pseudo_label][:self.n_repeat]
        dataset_x_data = torch.cat([dataset_x_data, dataset_x_data_novel])
        dataset_x_labels = torch.cat([dataset_x_labels, dataset_x_labels_novel])

        dataset_x = PoolDataset(dataset_x_data, None, dataset_x_labels)
        dataset_u = PoolDataset(None, dataset_u_data, None)

        dataloader_x = DataLoader(dataset_x, batch_size=self.batch_size // 2, shuffle=True, num_workers=4)
        dataloader_u = DataLoader(dataset_u, batch_size=self.batch_size // 2, shuffle=True, num_workers=4)
        self._show_data_info(dataset_x_labels.numpy())

        model_g.train()
        ce = CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_g.parameters(), lr=self.lr / 10, momentum=self.sgd_beta,
                                    weight_decay=self.sgd_decay)

        for epoch in range(1, self.epochs + 1):
            loss_list = []
            pred_list = []
            true_list = []

            for (_, x_data, x_labels), (_, u_data, _) in zip(dataloader_x, dataloader_u):
                loss_str = ""
                optimizer.zero_grad()

                x_weak = x_data
                u_weak = u_data
                u_strong = Augment.weak_augment(u_data)

                x_data = x_data.cuda()
                u_data = u_data.cuda()
                x_labels = x_labels.cuda().long()
                x_weak = x_weak.cuda()
                u_weak = u_weak.cuda()
                u_strong = u_strong.cuda()

                # ---- labeled data ----
                x_in_mask = torch.zeros_like(x_labels, dtype=torch.uint8)
                for k in self.centers.centers_index:
                    x_in_mask[x_labels == k] = 1

                f_x = self.model_f(x_data)[1]
                g_weak_x = model_g(x_weak)[1]
                x_out_mask = g_weak_x.max(dim=1)[0] >= self.tao
                x_out_mask[x_in_mask == 0] = True

                y_pred = g_weak_x.max(dim=1)[1]
                pred_list.extend(y_pred.cpu().tolist())
                true_list.extend(x_labels.cpu().tolist())

                # ls1
                ls1 = torch.tensor(0., dtype=torch.float32, requires_grad=True).cuda()
                if x_in_mask.max() == 1:
                    ls1 = ce(g_weak_x[x_in_mask], x_labels[x_in_mask])
                if x_out_mask.max() == 1:
                    ls1 = ls1 + ce(g_weak_x[x_out_mask], x_labels[x_out_mask])

                # ls2
                ls2 = torch.tensor(0., dtype=torch.float32, requires_grad=True).cuda()
                if self.alpha > 0.0 and x_in_mask.max() == 1:
                    p = f_x
                    q = g_weak_x + 1e-38
                    ls2 = torch.mean(p * torch.log(p / q))

                # ls
                loss_str += f"ls1 = {ls1:5.2f}    ls2 = {ls2:5.2f}"
                ls = ls1 + self.alpha * ls2

                # ---- unlabeled data ----
                lu = torch.tensor(0., dtype=torch.float32, requires_grad=True).cuda()
                if self.lamda_u > 0.0:
                    qu = model_g(u_weak)[1]
                    qu_cat = qu.argmax(dim=1)
                    tao_mask = qu.max(1)[0] >= self.tao
                    f_u = self.model_f(u_data)[1]
                    g_strong_u = model_g(u_strong)[1]

                    lu1 = torch.tensor(0., dtype=torch.float32, requires_grad=True).cuda()
                    lu2 = torch.tensor(0., dtype=torch.float32, requires_grad=True).cuda()
                    if tao_mask.max() == 1:
                        lu1 = ce(g_strong_u[tao_mask], qu_cat[tao_mask])
                        p = f_u[tao_mask]
                        q = g_strong_u[tao_mask] + 1e-38
                        lu2 = torch.mean(p * torch.log(p / q))
                        loss_str += f"    lu1 = {lu1:5.2f}    lu2 = {lu2:5.2f}"
                    lu = self.lamda_u * (lu1 + self.alpha * lu2)

                # total loss
                loss = ls + self.lamda_u * lu
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                loss_str = f"    loss = {loss:.2f}    " + loss_str
                self.logger.debug(loss_str)

            average_loss = np.array(loss_list, dtype=np.float32).mean()
            self.logger.debug(f"epoch = {epoch}    {average_loss}")
            acc = metrics.accuracy_score(y_pred=pred_list, y_true=true_list)
            self.logger.debug(f"Epoch {epoch:3d} / {self.epochs :<3d}    Accuracy = {acc * 100:5.2f}% ")
            m = metrics.confusion_matrix(y_pred=pred_list, y_true=true_list)
            self.logger.debug(f"\n{m}")
            if epoch == 5 or epoch >= 10:
                y_pred, y_true = self._predict(model_g)
                if epoch % 10 == 0:
                    model_g.save(os.path.join(self.log_folder, f"{self.dataset_name}_k{self.K}_g{epoch}.pth"))
                    self.logger.info(f"Saved Model_G with epoch = {epoch}.")
                    # self.logger.info(f"$ y_pred = {y_pred}")
                    # self.logger.info(f"$ y_true = {y_true}")

        return model_g

    def _predict(self, model_g: ResNet18) -> Tuple[list, list]:
        self._show_data_info(self._pool_true_labels.numpy())
        dataset = PoolDataset(self._pool_raw_data, self._pool_data, self._pool_true_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.model_f.eval()
        model_g.eval()

        y_pred = []
        y_true = []

        with torch.no_grad():
            for i, (batch_raw_data, batch_data, batch_labels) in enumerate(dataloader):
                batch_out = model_g(batch_data.cuda())[1].cpu()
                batch_pred = batch_out.argmax(dim=1)
                y_pred.extend(batch_pred.tolist())
                y_true.extend(batch_labels.tolist())

        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
        f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        m = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)
        self.logger.debug(f"Test Accuracy  = {accuracy * 100:5.2f}% ")
        self.logger.debug(f"Test Precision = {precision * 100:5.2f}% ")
        self.logger.debug(f"Test Recall    = {recall * 100:5.2f}% ")
        self.logger.debug(f"Test F1        = {f1 * 100:5.2f}% ")
        self.logger.debug(f"\n{m}")

        return y_pred, y_true
