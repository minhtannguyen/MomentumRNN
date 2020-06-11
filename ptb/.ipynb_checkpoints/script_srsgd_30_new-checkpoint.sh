#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU 0

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 2 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-2-seed-4" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 4 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-4-seed-4" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 10 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-10-seed-4" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 20 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-20-seed-4" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 30 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-30-seed-4" --gpu-id 0 &

# GPU 1

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 40 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-40-seed-4" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 50 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-50-seed-4" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 60 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-60-seed-4" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 70 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-70-seed-4" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 80 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-80-seed-4" --gpu-id 1 &

# GPU 2

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 90 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-90-seed-4" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 100 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-100-seed-4" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 150 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-150-seed-4" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 200 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-200-seed-4" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 250 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-250-seed-4" --gpu-id 2 &

# GPU 3

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 300 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-300-seed-4" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 350 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-350-seed-4" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 125 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-125-seed-4" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 6 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-6-seed-4" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.01 --restart 8 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 4 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.01-rs-8-seed-4" --gpu-id 3 &


wait 
echo "Done"
