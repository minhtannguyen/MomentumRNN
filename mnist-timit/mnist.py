import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os
import random
from torchvision import datasets, transforms

from parametrization import get_parameters
from orthogonal import OrthogonalRNN, OrthogonalMomentumRNN, OrthogonalAdamRNN, OrthogonalNesterovRNN
from trivializations import cayley_map, expm
from initialization import henaff_init_, cayley_init_

from momentumnet import MomentumLSTMCell, LSTMCell, NesterovLSTMCell, AdamLSTMCell

from utils import Logger
from tensorboardX import SummaryWriter
import fcntl

import time

parser = argparse.ArgumentParser(description='Exponential Layer MNIST Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=170)
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument('--lr_orth', type=float, default=7e-5)
parser.add_argument("--permute", action="store_true")
parser.add_argument("-m", "--mode",
                    choices=["exprnn", "dtriv", "cayley", "lstm", "mlstm", "mdtriv", "nlstm", "alstm", "adtriv", "ndtriv"],
                    default="dtriv",
                    type=str)
parser.add_argument('--K', type=str, default="100", help='The K parameter in the dtriv algorithm. It should be a positive integer or "infty".')
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="cayley",
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


args = parser.parse_args()

# logger
if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)
writer = SummaryWriter(os.path.join(args.checkpoint, 'tensorboard')) # write to tensorboard

logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='mnist-')

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

n_classes   = 10
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


class Model(nn.Module):
    def __init__(self, hidden_size, permute):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.permute = permute
        permute = np.random.RandomState(args.manualSeed)
        self.register_buffer("permutation", torch.LongTensor(permute.permutation(784)))
        if args.mode == "lstm":
            self.rnn = LSTMCell(1, hidden_size)
        elif args.mode == "mlstm":
            self.rnn = MomentumLSTMCell(1, hidden_size, mu=args.mu, epsilon=args.epsilon)
        elif args.mode == "alstm":
            self.rnn = AdamLSTMCell(1, hidden_size, mu=args.mu, epsilon=args.epsilon, mus=args.mus)
        elif args.mode == "nlstm":
            self.rnn = NesterovLSTMCell(1, hidden_size, epsilon=args.epsilon)
        elif args.mode == "mdtriv":
            self.rnn = OrthogonalMomentumRNN(1, hidden_size, initializer_skew=init, mode=mode, param=param, mu=args.mu, epsilon=args.epsilon)
        elif args.mode == "adtriv":
            self.rnn = OrthogonalAdamRNN(1, hidden_size, initializer_skew=init, mode=mode, param=param, mu=args.mu, epsilon=args.epsilon, mus=args.mus)
        elif args.mode == "ndtriv":
            self.rnn = OrthogonalNesterovRNN(1, hidden_size, initializer_skew=init, mode=mode, param=param, epsilon=args.epsilon)
        else:
            self.rnn = OrthogonalRNN(1, hidden_size, initializer_skew=init, mode=mode, param=param)

        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()


    def forward(self, inputs):
        if self.permute:
            inputs = inputs[:, self.permutation]

        if isinstance(self.rnn, OrthogonalRNN) or isinstance(self.rnn, OrthogonalMomentumRNN) or isinstance(self.rnn, OrthogonalAdamRNN) or isinstance(self.rnn, OrthogonalNesterovRNN):
            state = self.rnn.default_hidden(inputs[:, 0, ...])
        else:
            state = (torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device),
                     torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device))
            
        if isinstance(self.rnn, MomentumLSTMCell) or isinstance(self.rnn, NesterovLSTMCell):
            v = torch.zeros((inputs.size(0), 4 * self.hidden_size), device=inputs.device)
        elif isinstance(self.rnn, AdamLSTMCell):
            v = torch.zeros((inputs.size(0), 4 * self.hidden_size), device=inputs.device)
            s = torch.zeros((inputs.size(0), 4 * self.hidden_size), device=inputs.device)
        elif isinstance(self.rnn, OrthogonalAdamRNN):
            v = torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device)
            s = torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device)
        elif isinstance(self.rnn, OrthogonalMomentumRNN) or isinstance(self.rnn, OrthogonalNesterovRNN):
            v = torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device)
        
        iter_indx = 0
        for input in torch.unbind(inputs, dim=1):
            iter_indx = iter_indx + 1
            if isinstance(self.rnn, MomentumLSTMCell) or isinstance(self.rnn, OrthogonalMomentumRNN):
                out_rnn, state, v = self.rnn(input.unsqueeze(dim=1), state, v)
            elif isinstance(self.rnn, AdamLSTMCell) or isinstance(self.rnn, OrthogonalAdamRNN):
                out_rnn, state, v, s = self.rnn(input.unsqueeze(dim=1), state, v, s)
            elif isinstance(self.rnn, NesterovLSTMCell) or isinstance(self.rnn, OrthogonalNesterovRNN):
                out_rnn, state, v = self.rnn(input.unsqueeze(dim=1), state, v, k=iter_indx)
                if args.restart > 0 and not (iter_indx % args.restart):
                    iter_indx = 0
            else:
                out_rnn, state = self.rnn(input.unsqueeze(dim=1), state)
            
        return self.lin(out_rnn)

    def loss(self, logits, y):
        return self.loss_func(logits, y)

    def correct(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=1), y).float().sum()


def main():
    # Load data
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Model and optimizers
    model = Model(hidden_size, args.permute).to(device)
    model.train()

    if args.mode == "lstm" or args.mode == "mlstm" or args.mode == "nlstm" or args.mode == "alstm" or args.mode == "plstm":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
        optim_orth = None
    else:
        non_orth_params, log_orth_params = get_parameters(model)
        optim = torch.optim.RMSprop(non_orth_params, args.lr)
        optim_orth = torch.optim.RMSprop(log_orth_params, lr=args.lr_orth)

    best_test_acc = 0.
    iters = 0
    for epoch in range(epochs):
        processed = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device).view(-1, 784), batch_y.to(device)

            logits = model(batch_x)
            loss = model.loss(logits, batch_y)

            optim.zero_grad()
            if optim_orth:
                optim_orth.zero_grad()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            if optim_orth:
                optim_orth.step()

            with torch.no_grad():
                correct = model.correct(logits, batch_y)

            processed += len(batch_x)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%\tBest: {:.2f}%'.format(
                epoch, processed, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100 * correct/len(batch_x), best_test_acc))
            
            logger.file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%\tBest: {:.2f}%'.format(
                epoch, processed, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100 * correct/len(batch_x), best_test_acc))
            writer.add_scalars('train_loss', {args.mode: loss.item()}, iters)
            writer.add_scalars('train_acc', {args.mode: 100 * correct/len(batch_x)}, iters)
            iters += 1


        model.eval()
        with torch.no_grad():
            test_loss = 0.
            correct = 0.
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device).view(-1, 784), batch_y.to(device)
                logits = model(batch_x)
                test_loss += model.loss(logits, batch_y).float()
                correct += model.correct(logits, batch_y).float()

        test_loss /= len(test_loader)
        test_acc = 100 * correct / len(test_loader.dataset)
        best_test_acc = max(test_acc, best_test_acc)
        print()
        print("Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Best Accuracy: {:.2f}%"
                .format(test_loss, test_acc, best_test_acc))
        print()
        
        logger.file.write("Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Best Accuracy: {:.2f}%"
                .format(test_loss, test_acc, best_test_acc))
        writer.add_scalars('test_loss', {args.mode: test_loss}, epoch)
        writer.add_scalars('test_acc', {args.mode: test_acc}, epoch)

        model.train()
    
    logger.close()
    
    print('Best acc:')
    print(best_test_acc)
    
    with open("./all_results.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("%s\n"%args.checkpoint)
        f.write("best_acc %f\n\n"%best_test_acc)
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
