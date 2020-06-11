# Code for Experiments In the Paper "MomentumRNN: Integrating Momentum into Recurrent Neural Networks"

## Requirements
This code is tested inside the NVIDIA Pytorch docker container release 19.09. This container can be pulled from NVIDIA GPU Cloud as follows:

`docker pull nvcr.io/nvidia/pytorch:19.09-py3`

Detailed information on packages included in the NVIDIA Pytorch containter 19.09 can be found at [NVIDIA Pytorch Release 19.09](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-09.html#rel_19-09). In addition to those packages, the following packages are required:

- Sklearn: `pip install -U scikit-learn --user`
- OpenCV: `pip install opencv-python`
- Progress: `pip install progress`

In order to reproduce the plots in our papers, the following packages are needed:

- Pandas: `pip install pandas`
- Seaborn: `pip install seaborn`

To run our code without using the NVIDIA Pytorch containter, at least the following packages are required:

- Ubuntu 18.04 including Python 3.6 environment
- PyTorch 1.2.0
- NVIDIA CUDA 10.1.243 including cuBLAS 10.2.1.243
- NVIDIA cuDNN 7.6.3

## Training
A training recipe is provided for MNIST, PMNIST, and TIMIT experiments at `mnist-timit/recipes.md`. The recipe contains the commands to run experiments for reproducing Table 1, 2, 4, and 5 in our paper. 

Another training recipe is provided for Penn TreeBank experiments at `ptb/recipes.md`. The recipe contains the commands to run experiments for reproducing Table 3 in our paper. 

### A note on the notation
In our code, `mu`, `epsilon`, and `mus` are momentum, step size, and coefficients `beta` used for computing running averages of squared gradients in the paper.

## MomentumRNN
We provide implementation of MomentumLSTM, AdamLSTM, NesterovLSTM in `mnist-timit/momentumnet.py` and `ptb/momentumnet.py`.

We provide implementation of MomentumDTRIV, AdamDTRIV, and NesterovDTRIV in `mnist-timit/orthogonal.py` under the names OrthogonalMomentumRNN, OrthogonalAdamRNN, and OrthogonalNesterovRNN.

## A note on the PTB experiment
Run `getdata.sh` to acquire the Penn TreeBank datasets.

## A note on the TIMIT experiment
The TIMIT dataset is not open, but most universities and many other institutions have access to it.

To preprocess the data of the TIMIT dataset, we used the tools provided by Wisdom on the repository:

https://github.com/stwisdom/urnn

As mentioned in the repository, first downsample the TIMIT dataset using the `downsample_audio.m` present in the `matlab` folder.

> Downsample the TIMIT dataset to 8ksamples/sec using Matlab by running downsample_audio.m from the matlab directory. Make sure you modify the paths in `downsample_audio.m` for your system.

Create a `timit_data` folder to store all the files.

After that, modify the file `timit_prediction.py` and add the following lines after line 529.

    np.save("timit_data/lens_train.npy", lens_train)
    np.save("timit_data/lens_test.npy", lens_test)
    np.save("timit_data/lens_eval.npy", lens_eval)
    np.save("timit_data/train_x.npy", np.transpose(train_xdata, [1, 0, 2]))
    np.save("timit_data/train_z.npy", np.transpose(train_z, [1, 0, 2]))
    np.save("timit_data/test_x.npy",  np.transpose(test_xdata, [1, 0, 2]))
    np.save("timit_data/test_z.npy",  np.transpose(test_z, [1, 0, 2]))
    np.save("timit_data/eval_x.npy",  np.transpose(eval_xdata, [1, 0, 2]))
    np.save("timit_data/eval_z.npy",  np.transpose(eval_z, [1, 0, 2]))

Run this script to save the dataset in a format that can be loaded by the TIMIT dataset loader

    import numpy as np
    import torch

    train_x = torch.tensor(np.load('timit_data/train_x.npy'))
    train_y = torch.tensor(np.load('timit_data/train_z.npy'))
    lens_train = torch.tensor(np.load("timit_data/lens_train.npy"), dtype=torch.long)

    test_x = torch.tensor(np.load('timit_data/test_x.npy'))
    test_y = torch.tensor(np.load('timit_data/test_z.npy'))
    lens_test = torch.tensor(np.load("timit_data/lens_test.npy"), dtype=torch.long)

    val_x = torch.tensor(np.load('timit_data/eval_x.npy'))
    val_y = torch.tensor(np.load('timit_data/eval_z.npy'))
    lens_val = torch.tensor(np.load("timit_data/lens_eval.npy"), dtype=torch.long)

    training_set = (train_x, train_y, lens_train)
    test_set = (test_x, test_y, lens_test)
    val_set = (val_x, val_y, lens_val)
    with open("timit_data/training.pt", 'wb') as f:
        torch.save(training_set, f)
    with open("timit_data/test.pt", 'wb') as f:
        torch.save(test_set, f)
    with open("timit_data/val.pt", 'wb') as f:
        torch.save(val_set, f)
        
Finally, to run our code, place `training.pt`, `test.pt`, `val.pt` in /datasets/timit_data_trainNoSA_dev_coreTest

## Code for Plotting Figures in Our Paper
We provide code for plotting figures in our paper in the jupyter notebook `plot_code.ipynb`. Data needed for plotting is provided in `result4plot_final`. 