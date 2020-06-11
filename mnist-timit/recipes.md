# LSTM and Momentum-based LSTM trainings for MNIST classfication tasks. The following commands can be used to reproduce MNIST results in Table 1 in our paper.

```
CUDA_VISIBLE_DEVICES=0 python mnist.py -m "lstm" --hidden_size 128 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "mnist-lstm-n128-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "lstm" --hidden_size 256 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "mnist-lstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "mlstm" --hidden_size 128 --mu 0.6 --epsilon 0.6 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "mnist-momentumlstm-n128-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "mlstm" --hidden_size 256 --mu 0.6 --epsilon 0.6 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "mnist-momentumlstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "alstm" --hidden_size 256 --mu 0.6 --epsilon 0.6 --mus 0.1 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "mnist-adamlstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "alstm" --hidden_size 256 --mu 0.0 --epsilon 0.6 --mus 0.9 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "mnist-rmsproplstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "nlstm" --hidden_size 256 --mu 0.0 --epsilon 1.0 --restart 2 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "mnist-srlstm-n256-seed-0" --manualSeed 0 --gpu-id 0 
```

# LSTM and Momentum-based LSTM trainings for PMNIST classfication tasks. The following commands can be used to reproduce PMNIST results in Table 1 in our paper.

```
CUDA_VISIBLE_DEVICES=0 python mnist.py -m "lstm" --permute --hidden_size 128 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "pmnist-lstm-n128-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "lstm" --permute --hidden_size 256 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "pmnist-lstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "mlstm" --permute --hidden_size 128 --mu 0.6 --epsilon 1.0 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "pmnist-momentumlstm-n128-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "mlstm" --permute --hidden_size 256 --mu 0.6 --epsilon 1.0 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "pmnist-momentumlstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "alstm" --permute --hidden_size 256 --mu 0.6 --epsilon 1.0 --mus 0.01 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "pmnist-adamlstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "alstm" --permute --hidden_size 256 --mu 0.0 --epsilon 1.0 --mus 0.01 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "pmnist-rmsproplstm-n256-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "nlstm" --permute --hidden_size 256 --mu 0.0 --epsilon 0.9 --restart 40 --lr 0.001 --batch_size 128 --epochs 150 --checkpoint "pmnist-srlstm-n256-seed-0" --manualSeed 0 --gpu-id 0 
```

# LSTM and Momentum-based LSTM trainings for TIMIT speech prediction. The following commands can be used to reproduce TIMIT results in Table 2 in our paper.

```
CUDA_VISIBLE_DEVICES=0 python timit.py -m "lstm" --hidden_size 84 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-lstm-n84-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "lstm" --hidden_size 120 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-lstm-n120-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "lstm" --hidden_size 158 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-lstm-n158-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "mlstm" --hidden_size 84 --mu 0.3 --epsilon 0.1 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-momentumlstm-n84-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "mlstm" --hidden_size 120 --mu 0.3 --epsilon 0.1 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-momentumlstm-n120-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "mlstm" --hidden_size 158 --mu 0.3 --epsilon 0.1 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-momentumlstm-n158-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "alstm" --hidden_size 158 --mu 0.3 --epsilon 0.1 --mus 0.999 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-adamlstm-n158-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "alstm" --hidden_size 158 --mu 0.0 --epsilon 0.1 --mus 0.999 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-rmsproplstm-n158-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit.py -m "nlstm" --hidden_size 158 --mu 0.0 --epsilon 0.1 --restart 2 --lr 0.0001 --batch_size 32 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-srlstm-n158-seed-0" --manualSeed 0 --gpu-id 0

```

# DTRIV and Momentum-based DTRIV trainings for PMNIST classfication tasks. The following commands can be used to reproduce PMNIST results in Table 4 in our paper.

```
CUDA_VISIBLE_DEVICES=0 python mnist.py -m "dtriv" --permute --hidden_size 170 --K 1 --lr 0.0007 --lr_orth 0.0002 --batch_size 128 --epochs 150 --checkpoint "pmnist-dtriv-n170-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "dtriv" --permute --hidden_size 360 --K "infty" --lr 0.0007 --lr_orth 0.00005 --batch_size 128 --epochs 150 --checkpoint "pmnist-dtriv-n360-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "dtriv" --permute --hidden_size 512 --K "infty" --lr 0.0003 --lr_orth 0.00007 --batch_size 128 --epochs 150 --checkpoint "pmnist-dtriv-n512-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "mdtriv" --permute --hidden_size 170 --K 1 --mu 0.6 --epsilon 0.9 --lr 0.0007 --lr_orth 0.0002 --batch_size 128 --epochs 150 --checkpoint "pmnist-momentumdtriv-n170-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "mdtriv" --permute --hidden_size 360 --K "infty" --mu 0.3 --epsilon 0.3 --lr 0.0007 --lr_orth 0.00005 --batch_size 128 --epochs 150 --checkpoint "pmnist-momentumdtriv-n360-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python mnist.py -m "mdtriv" --permute --hidden_size 512 --K "infty" --mu 0.3 --epsilon 0.3 --lr 0.0003 --lr_orth 0.00007 --batch_size 128 --epochs 150 --checkpoint "pmnist-momentumdtriv-n512-seed-0" --manualSeed 0 --gpu-id 0
```

# DTRIV and Momentum-based DTRIV trainings for TIMIT speech prediction tasks. The following commands can be used to reproduce TIMIT results in Table 5 in our paper.

```
CUDA_VISIBLE_DEVICES=0 python timit_div.py -m "dtriv" --hidden_size 224 --K "infty" --lr 0.001 --lr_orth 0.0002 --batch_size 128 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-dtriv-n224-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit_div.py -m "dtriv" --hidden_size 322 --K "infty" --lr 0.001 --lr_orth 0.0002 --batch_size 128 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-dtriv-n322-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit_div.py -m "mdtriv" --hidden_size 224 --K "infty" --mu 0.3 --epsilon 0.1 --lr 0.001 --lr_orth 0.0002 --batch_size 128 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-dtriv-n224-seed-0" --manualSeed 0 --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python timit_div.py -m "mdtriv" --hidden_size 322 --K "infty" --mu 0.3 --epsilon 0.1 --lr 0.001 --lr_orth 0.0002 --batch_size 128 --epochs 700 --datadir '/datasets/timit_data_trainNoSA_dev_coreTest' --checkpoint "timit-dtriv-n322-seed-0" --manualSeed 0 --gpu-id 0
```