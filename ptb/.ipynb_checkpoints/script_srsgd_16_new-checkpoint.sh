#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU 0

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 2 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-2-seed-2" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 4 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-4-seed-2" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 10 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-10-seed-2" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 20 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-20-seed-2" --gpu-id 0 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 30 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-30-seed-2" --gpu-id 0 &

# GPU 1

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 40 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-40-seed-2" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 50 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-50-seed-2" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 60 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-60-seed-2" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 70 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-70-seed-2" --gpu-id 1 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 80 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-80-seed-2" --gpu-id 1 &

# GPU 2

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 90 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-90-seed-2" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 100 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-100-seed-2" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 150 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-150-seed-2" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 200 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-200-seed-2" --gpu-id 2 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 250 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-250-seed-2" --gpu-id 2 &

# GPU 3

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 300 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-300-seed-2" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 350 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-350-seed-2" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 125 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-125-seed-2" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 6 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-6-seed-2" --gpu-id 3 &

python main_momentum.py --model 'NLSTM' --epsilon 0.3 --restart 8 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-nlstmnew-eps-0.3-rs-8-seed-2" --gpu-id 3 &


wait 
echo "Done"
