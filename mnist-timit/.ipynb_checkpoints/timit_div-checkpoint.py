import torch
import torch.nn as nn
import numpy as np
import argparse
import datetime

import os
import random
import sys

from parametrization import get_parameters
from orthogonal import OrthogonalRNN, OrthogonalMomentumRNN, OrthogonalAdamRNN, OrthogonalNesterovRNN
from trivializations import cayley_map, expm
from initialization import henaff_init_, cayley_init_
from timit_loader import TIMIT

from momentumnet_forget_neg4 import MomentumLSTMCell, LSTMCell

from utils import Logger
from tensorboardX import SummaryWriter
import fcntl


parser = argparse.ArgumentParser(description='Exponential Layer TIMIT Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_orth', type=float, default=1e-4)
parser.add_argument("-m", "--mode",
                    choices=["exprnn", "dtriv", "cayley", "lstm", "mlstm", "mdtriv", "adtriv", "ndtriv"],
                    default="dtriv",
                    type=str)
parser.add_argument('--K', type=str, default="100", help='The K parameter in the dtriv algorithm. It should be a positive integer or "infty".')
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="henaff",
                    type=str)

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--mu', type=float, default=0.9)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--restart', type=int, default=0, help='restart momentum after this number of epoch')
parser.add_argument('--mus', type=float, default=0.9)

parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--datadir', default='datadir', type=str, metavar='PATH',
                    help='path that saves timit data (default: datadir)')


args = parser.parse_args()

# logger
if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)
writer = SummaryWriter(os.path.join(args.checkpoint, 'tensorboard')) # write to tensorboard

logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='timit-')

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

n_input     = 129
n_classes   = 129
batch_size  = args.batch_size
hidden_size = args.hidden_size
epochs      = args.epochs
device      = torch.device('cuda')

if args.init == "cayley":
    init =  cayley_init_
elif args.init == "henaff":
    init = henaff_init_

if args.K != "infty":
    args.K = int(args.K)
if args.mode == "exprnn":
    mode = "static"
    param = expm
elif args.mode == "dtriv" or args.mode == "mdtriv" or args.mode == "adtriv" or args.mode == "ndtriv":
    # We use 100 as the default to project back to the manifold.
    # This parameter does not really affect the convergence of the algorithms, even for K=1
    mode = ("dynamic", args.K, 100)
    param = expm
elif args.mode == "cayley":
    mode = "static"
    param = cayley_map


def masked_loss(lossfunc, logits, y, lens):
    """ Computes the loss of the first `lens` items in the batches """
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, l in enumerate(lens):
        mask[i, :l, :] = 1
    logits_masked = torch.masked_select(logits, mask)
    y_masked = torch.masked_select(y, mask)
    return lossfunc(logits_masked, y_masked)


class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        if args.mode == "lstm":
            self.rnn = LSTMCell(n_input, hidden_size)
        elif args.mode == "mlstm":
            self.rnn = MomentumLSTMCell(n_input, hidden_size, mu=args.mu, epsilon=args.epsilon)
        elif args.mode == "mdtriv":
            self.rnn = OrthogonalMomentumRNN(n_input, hidden_size, initializer_skew=init, mode=mode, param=param, mu=args.mu, epsilon=args.epsilon)
        elif args.mode == "adtriv":
            self.rnn = OrthogonalAdamRNN(n_input, hidden_size, initializer_skew=init, mode=mode, param=param, mu=args.mu, epsilon=args.epsilon, mus=args.mus)
        elif args.mode == "ndtriv":
            self.rnn = OrthogonalNesterovRNN(n_input, hidden_size, initializer_skew=init, mode=mode, param=param, epsilon=args.epsilon)
        else:
            self.rnn = OrthogonalRNN(n_input, hidden_size, initializer_skew=init, mode=mode, param=param)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.MSELoss()

    def forward(self, inputs):
        if isinstance(self.rnn, OrthogonalRNN) or isinstance(self.rnn, OrthogonalMomentumRNN) or isinstance(self.rnn, OrthogonalAdamRNN) or isinstance(self.rnn, OrthogonalNesterovRNN):
            state = self.rnn.default_hidden(inputs[:, 0, ...])
        else:
            state = (torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device),
                     torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device))
            
        if isinstance(self.rnn, MomentumLSTMCell):
            v = torch.zeros((inputs.size(0), 4 * self.hidden_size), device=inputs.device)
        elif isinstance(self.rnn, OrthogonalMomentumRNN) or isinstance(self.rnn, OrthogonalNesterovRNN):
            v = torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device)
        elif isinstance(self.rnn, OrthogonalAdamRNN):
            v = torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device)
            s = torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device)
            
        outputs = []
        iter_indx = 0
        for input in torch.unbind(inputs, dim=1):
            iter_indx = iter_indx + 1
            if isinstance(self.rnn, MomentumLSTMCell) or isinstance(self.rnn, OrthogonalMomentumRNN):
                out_rnn, state, v = self.rnn(input, state, v)
            elif isinstance(self.rnn, OrthogonalAdamRNN):
                out_rnn, state, v, s = self.rnn(input, state, v, s)
            elif isinstance(self.rnn, OrthogonalNesterovRNN):
                out_rnn, state, v = self.rnn(input, state, v, k=iter_indx)
                if args.restart > 0 and not (iter_indx % args.restart):
                    iter_indx = 0
            else:
                out_rnn, state = self.rnn(input, state)
            
            outputs.append(self.lin(out_rnn))
        return torch.stack(outputs, dim=1)

    def loss(self, logits, y, len_batch):
        return masked_loss(self.loss_func, logits, y, len_batch)


def main():
    # Load data
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        TIMIT(args.datadir, mode="train"),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Load test and val in one big batch
    test_loader = torch.utils.data.DataLoader(
        TIMIT(args.datadir, mode="test"),
        batch_size=400, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        TIMIT(args.datadir, mode="val"),
        batch_size=192, shuffle=True, **kwargs)


    # Model and optimizers
    model = Model(hidden_size).to(device)
    model.train()

    if args.mode == "lstm" or args.mode == "mlstm":
        #optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.1, momentum=0.9)
        optim = torch.optim.Adam(model.parameters(), args.lr)
        optim_orth = None
    else:
        non_orth_params, log_orth_params = get_parameters(model)
        optim = torch.optim.Adam(non_orth_params, args.lr)
        optim_orth = torch.optim.RMSprop(log_orth_params, lr=args.lr_orth)

    best_test = 1e7
    best_validation = 1e7
    iters = 0

    for epoch in range(epochs):
        init_time = datetime.datetime.now()
        processed = 0
        step = 1
        for batch_idx, (batch_x, batch_y, len_batch) in enumerate(train_loader):
            batch_x, batch_y, len_batch = batch_x.to(device), batch_y.to(device), len_batch.to(device)

            logits = model(batch_x)
            loss = model.loss(logits, batch_y, len_batch)

            optim.zero_grad()
            if optim_orth:
                optim_orth.zero_grad()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            if optim_orth:
                optim_orth.step()

            processed += len(batch_x)
            step += 1

            print("Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} "
                  .format(epoch, processed, len(train_loader.dataset),
                      100. * processed / len(train_loader.dataset), loss))
            
            logger.file.write("Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} "
                  .format(epoch, processed, len(train_loader.dataset),
                      100. * processed / len(train_loader.dataset), loss))
            
            writer.add_scalars('train_loss', {args.mode: loss.item()}, iters)
            iters += 1

        model.eval()
        with torch.no_grad():
            # There's just one batch for test and validation
            for batch_x, batch_y, len_batch in test_loader:
                batch_x, batch_y, len_batch = batch_x.to(device), batch_y.to(device), len_batch.to(device)
                logits = model(batch_x)
                loss_test = model.loss(logits, batch_y, len_batch)

            for batch_x, batch_y, len_batch in val_loader:
                batch_x, batch_y, len_batch = batch_x.to(device), batch_y.to(device), len_batch.to(device)
                logits = model(batch_x)
                loss_val = model.loss(logits, batch_y, len_batch)

            if loss_val < best_validation:
                best_validation = loss_val
                bestval_test = loss_test
                state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': best_validation,
                'optimizer' : optim.state_dict(),}
                filepath = os.path.join(args.checkpoint, 'model_best_val.pth.tar')
                torch.save(state, filepath)
                
            if loss_test < best_test:
                best_test = loss_test
                besttest_val = loss_val
                state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': best_test,
                'optimizer' : optim.state_dict(),}
                filepath = os.path.join(args.checkpoint, 'model_best_test.pth.tar')
                torch.save(state, filepath)

        print()
        print("Val:  Loss: {:.2f}\tBest: {:.2f}\tBestVal_Test: {:.2f}".format(loss_val, best_validation, bestval_test))
        print("Test: Loss: {:.2f}\tBest: {:.2f}\tBestTest_Val: {:.2f}".format(loss_test, best_test, besttest_val))
        
        logger.file.write("Val:  Loss: {:.2f}\tBest: {:.2f}\tBestVal_Test: {:.2f}".format(loss_val, best_validation, bestval_test))
        logger.file.write("Test: Loss: {:.2f}\tBest: {:.2f}\tBestTest_Val: {:.2f}".format(loss_test, best_test, besttest_val))
        
        writer.add_scalars('val_loss', {args.mode: loss_val}, epoch)
        writer.add_scalars('test_loss', {args.mode: loss_test}, epoch)
        
        print()

        model.train()
        
    logger.close()
    
    print("BestVal: {:.2f} \tBestVal_Test: {:.2f}".format(best_validation, bestval_test))
    print("BestTest: {:.2f}\tBestTest_Val: {:.2f}".format(best_test, besttest_val))
    
    with open("./all_results_timit.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("%s\n"%args.checkpoint)
        f.write("BestVal: {:.2f} \tBestVal_Test: {:.2f}\n".format(best_validation, bestval_test))
        f.write("BestTest: {:.2f}\tBestTest_Val: {:.2f}\n\n".format(best_test, besttest_val))
        fcntl.flock(f, fcntl.LOCK_UN)

if __name__ == "__main__":
    main()
