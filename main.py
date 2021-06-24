import os
import argparse

import stream_train
from Logger import set_logger
from Modules import ResNet34, Centers, Memory
from init_train import load_or_init_train

if __name__ == '__main__':
    # global arguments
    parser = argparse.ArgumentParser(description="S2OSC")
    parser.add_argument("--dataset", type=str, default="c10", choices=["m", "fm", "c10", "cinic", "svhn"])
    parser.add_argument("--device", type=str, default="0", choices=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--train", type=bool, default=True, help="True to train, False to load pretrained files.")
    parser.add_argument("--K", type=int, default=300, help="Number of instances for each class stores in the memory.")
    parser.add_argument("--M", type=float, default=2000)
    parser.add_argument("--T", type=float, default=3)
    parser.add_argument("--lamda", type=float, default=1.0)
    # initial arguments
    parser.add_argument("--init_epochs", type=int, default=20)
    parser.add_argument("--init_batch_size", type=int, default=64)
    parser.add_argument("--init_lr", type=float, default=0.01)
    parser.add_argument("--init_SGD_beta", type=float, default=0.9)
    parser.add_argument("--init_SGD_decay", type=float, default=0.001)
    # stream arguments
    parser.add_argument("--stream_epochs", type=int, default=30)
    parser.add_argument("--stream_batch_size", type=int, default=64)
    parser.add_argument("--stream_lr", type=float, default=0.005)
    parser.add_argument("--stream_SGD_beta", type=float, default=0.9)
    parser.add_argument("--stream_SGD_decay", type=float, default=0.0005)
    # loss function arguments
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--lamda_u", type=float, default=0.2)
    parser.add_argument("--tao", type=float, default=0.85)

    args = parser.parse_args()

    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # set logger
    logger, log_folder = set_logger(name=f"{args.dataset}")
    logger.info(args)

    # initial train
    n_out = 6
    modelF = ResNet34(n_out=n_out, T=args.T).cuda()
    memory = Memory(K=args.K)
    centers = Centers(rate_old=0.8, n_centers=n_out)
    load_or_init_train(logger=logger, dataset_name=args.dataset, train=args.train, n_out=n_out,
                       model=modelF, memory=memory, centers=centers, running_path=log_folder,
                       epochs=args.init_epochs, batch_size=args.init_batch_size, lr=args.init_lr,
                       sgd_beta=args.init_SGD_beta, sgd_decay=args.init_SGD_decay)

    # stream train
    stream = stream_train.Stream(logger=logger, log_folder=log_folder, dataset_name=args.dataset,
                                 model_f=modelF, centers=centers, memory=memory, T=args.T, K=args.K, n_out=n_out,
                                 epochs=args.stream_epochs, batch_size=args.stream_batch_size, lr=args.stream_lr,
                                 sgd_beta=args.stream_SGD_beta, sgd_decay=args.stream_SGD_decay,
                                 lamda=args.lamda, alpha=args.alpha, lamda_u=args.lamda_u, tao=args.tao)
    stream.train()
