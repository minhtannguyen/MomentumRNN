import torch
import torch.nn as nn
import torch.jit as jit
from torch.nn import Parameter
from torch.nn import functional as F

from parametrization import Parametrization

import math

import time

class MomentumLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mu, epsilon, bias=True, fg_init=1.0):
        super(MomentumLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fg_init = fg_init
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        # for momentumnet
        self.mu = mu
        self.epsilon = epsilon
        
        self.reset_parameters(hidden_size)

    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.x2h.weight)
        nn.init.eye_(self.h2h.weight)
        nn.init.zeros_(self.x2h.bias)
        self.x2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init) 
        nn.init.zeros_(self.h2h.bias)
        self.h2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init)
    
    def forward(self, x, hidden, v):
        
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        v = v.view(-1, v.size(1))
        
        vy = self.mu * v + self.epsilon * self.x2h(x)
        
        gates = vy + self.h2h(hx)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return hy, (hy, cy), vy
    
    
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, fg_init=1.0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fg_init = fg_init
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters(hidden_size)

    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.x2h.weight)
        nn.init.eye_(self.h2h.weight)
        nn.init.zeros_(self.x2h.bias)
        self.x2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init) 
        nn.init.zeros_(self.h2h.bias)
        self.h2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init) 
        
    def forward(self, x, hidden):
        
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return hy, (hy, cy)
    
    
class NesterovLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, epsilon, bias=True, fg_init=1.0):
        super(NesterovLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fg_init = fg_init
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        # for momentumnet
        self.epsilon = epsilon
        
        self.reset_parameters(hidden_size)

    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.x2h.weight)
        nn.init.eye_(self.h2h.weight)
        nn.init.zeros_(self.x2h.bias)
        self.x2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init) 
        nn.init.zeros_(self.h2h.bias)
        self.h2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init)
    
    def forward(self, x, hidden, v, k):
        
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        v = v.view(-1, v.size(1))
        
        vy = self.h2h(hx) - self.epsilon * self.x2h(x)
        
        gates = vy + (k-1.0)/(k+2.0) * (vy - v)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return hy, (hy, cy), vy

class AdamLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mu, epsilon, mus, bias=True, fg_init=1.0):
        super(AdamLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fg_init = fg_init
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        # for momentumnet
        self.mu = mu
        self.epsilon = epsilon
        self.mus = mus
        
        self.reset_parameters(hidden_size)

    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.x2h.weight)
        nn.init.eye_(self.h2h.weight)
        nn.init.zeros_(self.x2h.bias)
        self.x2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init) 
        nn.init.zeros_(self.h2h.bias)
        self.h2h.bias.data[hidden_size:(2 * hidden_size)].fill_(self.fg_init)
    
    def forward(self, x, hidden, v, s):
        
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        v = v.view(-1, v.size(1))
        s = s.view(-1, v.size(1))
        
        grad_val = self.x2h(x)
        
        vy = self.mu * v + self.epsilon * grad_val
        sy = self.mus * s + (1.0 - self.mus) * (grad_val * grad_val)
        
        gates = vy/torch.sqrt(sy + 1e-16) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return hy, (hy, cy), vy, sy