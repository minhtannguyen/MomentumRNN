#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 0 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-lstm-seed-0" --gpu-id 0 &

python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 1 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-lstm-seed-1" --gpu-id 0 &

python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 2 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-lstm-seed-2" --gpu-id 0 &

python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --manualSeed 3 --epoch 500 --save PTB.pt --checkpoint "/tanresults2/experiments-momentumnet/ptb-lstm-seed-3" --gpu-id 0 &


wait 
echo "Done"