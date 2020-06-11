#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU 1

python main_momentum_temp.py --model 'MLSTM' --mu 0.99 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-1.0-seed-0" --gpu-id 1 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.99 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-1.0-seed-1" --gpu-id 1 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.99 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-1.0-seed-2" --gpu-id 1 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.99 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-1.0-seed-3" --gpu-id 1 &

# GPU 2

python main_momentum_temp.py --model 'MLSTM' --mu 0.9 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-1.0-seed-0" --gpu-id 2 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.9 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-1.0-seed-1" --gpu-id 2 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.9 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-1.0-seed-2" --gpu-id 2 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.9 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-1.0-seed-3" --gpu-id 2 &

# GPU 3

python main_momentum_temp.py --model 'MLSTM' --mu 0.6 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-1.0-seed-0" --gpu-id 3 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.6 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-1.0-seed-1" --gpu-id 3 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.6 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-1.0-seed-2" --gpu-id 3 &

python main_momentum_temp.py --model 'MLSTM' --mu 0.6 --epsilon 1.0 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-1.0-seed-3" --gpu-id 3 &


wait 
echo "Done"