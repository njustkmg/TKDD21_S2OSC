# S2OSC

This is code for the paper "A Holistic Semi-Supervised Approach for Open Set Classification".

# Dependencies
Our code is based on the following platform and packages.
- Python (3.7.0)
- pillow (5.3.0)
- numpy (1.17.2)
- pytorch (1.1.0)
- torchvision (0.3.0)
- scikit-learn (0.21.3)

# Data Preparation
First, you need to generate data. Run `generate_data.py` to create data for each dataset. Take CIFAR10 for example:

```
python generate_data.py --dataset c10 --path XXX
```

The created data files will be stored in `./DataSets/XXX/` by default. `XXX_init.npy` is the initial known data and `XXX_stream.npy` is the novel data mixed with the known data. All data are stored in the numpy format. The file structure looks like this:

```
ROOT
│
├────main.py
├────generate_data.py
├────......
├────......
│
└────DataSets
     ├────CIFAR10
     │    ├────cifar10_init.npy
     │    └────cifar10_stream.npy
     └────......
```

# Run S2OSC
You just need to run `main.py`. This file includes initial training for Model_F and stream training for Model_G. You may want to specify paramters for `dataset` and `device`.

```
python main.py --dataset c10 --device 0
```