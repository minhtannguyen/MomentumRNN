import torch
import torch.nn as nn
import torch.jit as jit
from torch.nn import Parameter
from torch.nn import functional as F

from parametrization import Parametrization

import math

import time

class MomentumLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, mu, epsilon, bias=True):
        super(MomentumLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        # for momentumnet
        self.mu = mu
        self.epsilon = epsilon
        
        self.reset_parameters(hidden_size)
            
    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.eye_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        self.bias_ih.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        nn.init.zeros_(self.bias_hh)
        self.bias_hh.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        
    def lstmcell(self, x, hidden, v):
        hx, cx = hidden
        
        hx = hx.squeeze() if hx.shape[1] > 1 else hx[0]
        cx = cx.squeeze() if cx.shape[1] > 1 else cx[0]
    
        x = x.view(-1, x.size(1))
        v = v.squeeze() if v.shape[1] > 1 else v[0]
        
        vy = self.mu * v + self.epsilon * (torch.mm(x, self.weight_ih.t()) + self.bias_ih)
        gates = vy + (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        
        # gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        cy = cy.unsqueeze(0)
        hy = hy.unsqueeze(0)
        vy = vy.unsqueeze(0)
        
        return hy, (hy, cy), vy 
        
    def forward(self, input_, hidden=None, v=None):
        # input_ is of dimensionalty (#batch, time, input_size, ...)
        outputs = []
        for x in torch.unbind(input_, dim=0):
            out_rnn, hidden, v = self.lstmcell(x, hidden, v)
            if out_rnn.shape[1] > 1:
                outputs.append(out_rnn.squeeze())
            else:
                outputs.append(out_rnn[0])
        
        return torch.stack(outputs, dim=0), hidden, v
    
    
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters(hidden_size)
            
    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.eye_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        self.bias_ih.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        nn.init.zeros_(self.bias_hh)
        self.bias_hh.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        
    def lstmcell(self, x, hidden):
        
        hx, cx = hidden
        
        hx = hx.squeeze() if hx.shape[1] > 1 else hx[0]
        cx = cx.squeeze() if cx.shape[1] > 1 else cx[0]
        
        x = x.view(-1, x.size(1))
        
        gates = torch.mm(x, self.weight_ih.t()) + torch.mm(hx, self.weight_hh.t()) + self.bias_ih + self.bias_hh
        
        # gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        cy = cy.unsqueeze(0)
        hy = hy.unsqueeze(0)
        
        return hy, (hy, cy) 
        
    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (#batch, time, input_size, ...)
        outputs = []
        for x in torch.unbind(input_, dim=0):
            out_rnn, hidden = self.lstmcell(x, hidden)
            if out_rnn.shape[1] > 1:
                outputs.append(out_rnn.squeeze())
            else:
                outputs.append(out_rnn[0])
        
        return torch.stack(outputs, dim=0), hidden
    
    
class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters(hidden_size)
            
    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.eye_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        self.bias_ih.data[0:hidden_size].fill_(1.0) 
        nn.init.zeros_(self.bias_hh)
        self.bias_hh.data[0:hidden_size].fill_(1.0) 
        
    def grucell(self, x, hidden):
        
        hidden = hidden.squeeze() if hidden.shape[1] > 1 else hidden[0]
        
        x = x.view(-1, x.size(1))
        
        gi = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        gh = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        
        hy = hy.unsqueeze(0)

        return hy, hy
        
    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (#batch, time, input_size, ...)
        outputs = []
        for x in torch.unbind(input_, dim=0):
            out_rnn, hidden = self.grucell(x, hidden)
            if out_rnn.shape[1] > 1:
                outputs.append(out_rnn.squeeze())
            else:
                outputs.append(out_rnn[0])
        
        return torch.stack(outputs, dim=0), hidden
    
    
class AdamLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, mu, epsilon, mus, bias=True):
        super(AdamLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        # for momentumnet
        self.mu = mu
        self.epsilon = epsilon
        self.mus = mus
        
        self.reset_parameters(hidden_size)
            
    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.eye_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        self.bias_ih.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        nn.init.zeros_(self.bias_hh)
        self.bias_hh.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        
    def lstmcell(self, x, hidden, v, s):
        hx, cx = hidden
        
        hx = hx.squeeze() if hx.shape[1] > 1 else hx[0]
        cx = cx.squeeze() if cx.shape[1] > 1 else cx[0]
    
        x = x.view(-1, x.size(1))
        v = v.squeeze() if v.shape[1] > 1 else v[0]
        s = s.squeeze() if s.shape[1] > 1 else s[0]
        
        grad_val = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        
        vy = self.mu * v + self.epsilon * grad_val
        sy = self.mus * s + (1.0 - self.mus) * (grad_val * grad_val)
        
        gates = vy/torch.sqrt(sy + 1e-16) + (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        
        # gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        cy = cy.unsqueeze(0)
        hy = hy.unsqueeze(0)
        vy = vy.unsqueeze(0)
        sy = sy.unsqueeze(0)
        
        return hy, (hy, cy), vy, sy 
        
    def forward(self, input_, hidden=None, v=None, s=None):
        # input_ is of dimensionalty (#batch, time, input_size, ...)
        outputs = []
        for x in torch.unbind(input_, dim=0):
            out_rnn, hidden, v, s = self.lstmcell(x, hidden, v, s)
            if out_rnn.shape[1] > 1:
                outputs.append(out_rnn.squeeze())
            else:
                outputs.append(out_rnn[0])
        
        return torch.stack(outputs, dim=0), hidden, v, s
    
    
class NesterovLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, epsilon, restart, bias=True):
        super(NesterovLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        # for momentumnet
        self.epsilon = epsilon
        self.restart = restart
        
        self.reset_parameters(hidden_size)
            
    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.eye_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        self.bias_ih.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        nn.init.zeros_(self.bias_hh)
        self.bias_hh.data[hidden_size:(2 * hidden_size)].fill_(1.0) 
        
    def lstmcell(self, x, hidden, v, k):
        hx, cx = hidden
        
        hx = hx.squeeze() if hx.shape[1] > 1 else hx[0]
        cx = cx.squeeze() if cx.shape[1] > 1 else cx[0]
    
        x = x.view(-1, x.size(1))
        v = v.squeeze() if v.shape[1] > 1 else v[0]
        
        vy = (k-1.0)/(k+2.0) * v + self.epsilon * (torch.mm(x, self.weight_ih.t()) + self.bias_ih)
        gates = vy + (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        
        # gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        cy = cy.unsqueeze(0)
        hy = hy.unsqueeze(0)
        vy = vy.unsqueeze(0)
        
        return hy, (hy, cy), vy 
        
    def forward(self, input_, hidden=None, v=None):
        # input_ is of dimensionalty (#batch, time, input_size, ...)
        outputs = []
        iter_indx = 0
        for x in torch.unbind(input_, dim=0):
            iter_indx = iter_indx + 1
            out_rnn, hidden, v = self.lstmcell(x, hidden, v, k=iter_indx)
            if self.restart > 0 and not (iter_indx % self.restart):
                iter_indx = 0
            if out_rnn.shape[1] > 1:
                outputs.append(out_rnn.squeeze())
            else:
                outputs.append(out_rnn[0])
        
        return torch.stack(outputs, dim=0), hidden, v