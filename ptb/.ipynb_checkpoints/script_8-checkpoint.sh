#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU 0

python main_momentum.py --model 'MLSTM' --mu 0.0 --epsilon 0.3 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.3-seed-0" --gpu-id 0 &

python main_momentum.py --model 'MLSTM' --mu 0.0 --epsilon 0.3 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.3-seed-1" --gpu-id 0 &

python main_momentum.py --model 'MLSTM' --mu 0.0 --epsilon 0.3 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.3-seed-2" --gpu-id 0 &

python main_momentum.py --model 'MLSTM' --mu 0.0 --epsilon 0.3 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.3-seed-3" --gpu-id 0 &

# GPU 1

python main_momentum.py --model 'MLSTM' --mu 0.99 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-0.1-seed-0" --gpu-id 1 &

python main_momentum.py --model 'MLSTM' --mu 0.99 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-0.1-seed-1" --gpu-id 1 &

python main_momentum.py --model 'MLSTM' --mu 0.99 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-0.1-seed-2" --gpu-id 1 &

python main_momentum.py --model 'MLSTM' --mu 0.99 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.99-eps-0.1-seed-3" --gpu-id 1 &

# GPU 2

python main_momentum.py --model 'MLSTM' --mu 0.9 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-0.1-seed-0" --gpu-id 2 &

python main_momentum.py --model 'MLSTM' --mu 0.9 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-0.1-seed-1" --gpu-id 2 &

python main_momentum.py --model 'MLSTM' --mu 0.9 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-0.1-seed-2" --gpu-id 2 &

python main_momentum.py --model 'MLSTM' --mu 0.9 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.9-eps-0.1-seed-3" --gpu-id 2 &

# GPU 3

python main_momentum.py --model 'MLSTM' --mu 0.6 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-0.1-seed-0" --gpu-id 3 &

python main_momentum.py --model 'MLSTM' --mu 0.6 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-0.1-seed-1" --gpu-id 3 &

python main_momentum.py --model 'MLSTM' --mu 0.6 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-0.1-seed-2" --gpu-id 3 &

python main_momentum.py --model 'MLSTM' --mu 0.6 --epsilon 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-0.1-seed-3" --gpu-id 3 &

wait 
echo "Done"
