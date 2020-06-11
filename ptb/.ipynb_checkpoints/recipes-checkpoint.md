# LSTM and Momentum-based LSTM trainings for Penn TreeBank language modeling task at word level. The following commands can be used to reproduce results in Table 3 in our paper.

```
CUDA_VISIBLE_DEVICES=0 python main_momentum.py --model 'LSTM' --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "ptb-lstm-seed-0" --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python main_momentum.py --model 'MLSTM' --mu 0.0 --epsilon 0.6 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-momentumlstm-seed-0" --gpu-id 0
```
