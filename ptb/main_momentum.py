import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

import sys
import os
import random

import data
import model_momentum

from utils import batchify, get_batch, repackage_hidden

from monitor_utils import Logger
from tensorboardX import SummaryWriter
import fcntl

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU, MLSTM, ALSTM, NLSTM)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--manualSeed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--mu', type=float, default=0.9)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--mus', type=float, default=0.999)
parser.add_argument('--restart', type=int, default=0, help='restart momentum after this number of epoch')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()
args.tied = True

# logger
if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)
writer = SummaryWriter(os.path.join(args.checkpoint, 'tensorboard')) # write to tensorboard

logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='ptb-')

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    
iters = 0

"""
Load data
##############################################################################
"""


def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
# ##############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model_momentum.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, mu=args.mu, mus=args.mus, epsilon=args.epsilon, restart=args.restart)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
# ##############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    if args.model == 'MLSTM' or args.model == 'NLSTM':
        hidden, v = model.init_hidden(batch_size)
    elif args.model == 'ALSTM':
        hidden, v, s = model.init_hidden(batch_size)
    else:
        hidden = model.init_hidden(batch_size)
        
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        if args.model == 'MLSTM' or args.model == 'NLSTM':
            output, hidden, v = model(data, hidden, v)
        elif args.model == 'ALSTM':
            output, hidden, v, s = model(data, hidden, v, s)
        else:
            output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
        if args.model == 'MLSTM' or args.model == 'NLSTM':
            v = repackage_hidden(v)
        if args.model == 'ALSTM':
            v = repackage_hidden(v)
            s = repackage_hidden(s)
    return total_loss.item() / len(data_source)


def train():
    global iters
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model == 'MLSTM' or args.model == 'NLSTM':
        hidden, v = model.init_hidden(args.batch_size)
    elif args.model == 'ALSTM':
        hidden, v, s = model.init_hidden(args.batch_size)
    else:
        hidden = model.init_hidden(args.batch_size)
        
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        if args.model == 'MLSTM' or args.model == 'NLSTM':
            v = repackage_hidden(v)
        if args.model == 'ALSTM':
            v = repackage_hidden(v)
            s = repackage_hidden(s)
        optimizer.zero_grad()
        
        if args.model == 'MLSTM' or args.model == 'NLSTM':
            output, hidden, v, rnn_hs, dropped_rnn_hs = model(data, hidden, v, return_h=True)
        elif args.model == 'ALSTM':
            output, hidden, v, s, rnn_hs, dropped_rnn_hs = model(data, hidden, v, s, return_h=True)
        else:
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        
        writer.add_scalars('train_total_loss', {args.model: loss.data.item()}, iters)
        writer.add_scalars('train_raw_loss', {args.model: raw_loss.data.item()}, iters)
        writer.add_scalars('train_ppl', {args.model: math.exp(raw_loss.data.item())}, iters)
        writer.add_scalars('train_bpc', {args.model: raw_loss.data.item()/math.log(2)}, iters)
        
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            
            logger.file.write('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}\n'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
        iters += 1

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            writer.add_scalars('val_loss', {args.model: val_loss2}, epoch)
            writer.add_scalars('val_ppl', {args.model: math.exp(val_loss2)}, epoch)
            writer.add_scalars('val_bpc', {args.model: val_loss2 / math.log(2)}, epoch)
            
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)
            
            logger.file.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}\n'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))

            if val_loss2 < stored_loss:
                model_save(os.path.join(args.checkpoint, args.save))
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            
            writer.add_scalars('val_loss', {args.model: val_loss}, epoch)
            writer.add_scalars('val_ppl', {args.model: math.exp(val_loss)}, epoch)
            writer.add_scalars('val_bpc', {args.model: val_loss / math.log(2)}, epoch)
            
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            
            logger.file.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}\n'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))

            if val_loss < stored_loss:
                model_save(os.path.join(args.checkpoint, args.save))
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(os.path.join(args.checkpoint, args.save), epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(os.path.join(args.checkpoint, args.save))

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)

print('=' * 89)
print('| End of training | val loss {:5.2f} | val ppl {:8.2f} | val bpc {:8.3f}'.format(
    stored_loss, math.exp(stored_loss), stored_loss / math.log(2)))
print('=' * 89)

logger.file.write('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}\n'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))

logger.file.write('| End of training | val loss {:5.2f} | val ppl {:8.2f} | val bpc {:8.3f}\n'.format(
    stored_loss, math.exp(stored_loss), stored_loss / math.log(2)))

logger.close()

with open("./all_results.txt", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("%s\n"%args.checkpoint)
    f.write('test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}\n'.format(test_loss, math.exp(test_loss), test_loss / math.log(2)))
    f.write('valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}\n\n'.format(stored_loss, math.exp(stored_loss), stored_loss / math.log(2)))
    fcntl.flock(f, fcntl.LOCK_UN)
