#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU 0

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.999 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.999-seed-4" --gpu-id 0 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.99 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.99-seed-4" --gpu-id 0 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.9 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.9-seed-4" --gpu-id 0 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.8 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.8-seed-4" --gpu-id 0 &

# GPU 1

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.7 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.7-seed-4" --gpu-id 1 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.6 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.6-seed-4" --gpu-id 1 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.5 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.5-seed-4" --gpu-id 1 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.4 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.4-seed-4" --gpu-id 1 &

# GPU 2

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.3 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.3-seed-4" --gpu-id 2 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.2 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.2-seed-4" --gpu-id 2 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.1-seed-4" --gpu-id 2 &

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.01 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.01-seed-4" --gpu-id 2 &

# GPU 3

python main_momentum.py --model 'ALSTM' --mu 0.0 --epsilon 0.6 --mus 0.001 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.0-eps-0.6-mus-0.001-seed-4" --gpu-id 3 &

python main_momentum.py --model 'ALSTM' --mu 0.6 --epsilon 0.6 --mus 0.01 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-0.6-mus-0.01-seed-4" --gpu-id 3 &

python main_momentum.py --model 'ALSTM' --mu 0.6 --epsilon 0.6 --mus 0.1 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-0.6-mus-0.1-seed-4" --gpu-id 3 &

python main_momentum.py --model 'ALSTM' --mu 0.6 --epsilon 0.6 --mus 0.3 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-mlstm-mu-0.6-eps-0.6-mus-0.3-seed-4" --gpu-id 3 &


wait 
echo "Done"
