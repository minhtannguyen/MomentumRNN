#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU 0

python -u main_momentum_temp.py --model 'MLSTM' --mu 0.0 --epsilon 0.1 --epochs 500 --nlayers 3 --emsize 200 --nhid 1000 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.25 --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 1.2e-6 --bptt 150 --batch_size 128 --optimizer adam --lr 2e-3 --data data/pennchar --save PTBC.pt --when 300 400 --checkpoint "/tanresults2/experiments-momentumnet/ptb-char-mlstm-mu-0.0-eps-0.1-seed-0" --manualSeed 0 --gpu-id 2 &

python -u main_momentum_temp.py --model 'MLSTM' --mu 0.0 --epsilon 0.1 --epochs 500 --nlayers 3 --emsize 200 --nhid 1000 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.25 --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 1.2e-6 --bptt 150 --batch_size 128 --optimizer adam --lr 2e-3 --data data/pennchar --save PTBC.pt --when 300 400 --checkpoint "/tanresults2/experiments-momentumnet/ptb-char-mlstm-mu-0.0-eps-0.1-seed-1" --manualSeed 1 --gpu-id 2 &




wait 
echo "Done"
