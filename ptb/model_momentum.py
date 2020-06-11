import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import warnings
warnings.filterwarnings('ignore')

from momentumnet import LSTM, MomentumLSTM, GRU, AdamLSTM, NesterovLSTM

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, mu=0.9, epsilon=0.1, mus=0.999, restart=0):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU', 'MLSTM', 'NLSTM', 'ALSTM'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), bias=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'MLSTM':
            self.rnns = [MomentumLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), mu=mu, epsilon=epsilon, bias=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'ALSTM':
            self.rnns = [AdamLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), mu=mu, epsilon=epsilon, mus=mus, bias=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'NLSTM':
            self.rnns = [NesterovLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), epsilon=epsilon, restart=restart, bias=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, bias=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.mu = mu
        self.epsilon = epsilon
        self.mus = mus
        self.restart = restart
        

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, v=None, s=None, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)
        
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        new_velocity = []
        new_scale = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            if self.rnn_type == 'MLSTM' or self.rnn_type == 'NLSTM':
                raw_output, new_h, new_v = rnn(raw_output, hidden[l], v[l])
            elif self.rnn_type == 'ALSTM':
                raw_output, new_h, new_v, new_s = rnn(raw_output, hidden[l], v[l], s[l])
            else:
                raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            if self.rnn_type == 'MLSTM' or self.rnn_type == 'NLSTM':
                new_velocity.append(new_v)
            if self.rnn_type == 'ALSTM':
                new_velocity.append(new_v)
                new_scale.append(new_s)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        if self.rnn_type == 'MLSTM' or self.rnn_type == 'NLSTM':
            v = new_velocity
            
        if self.rnn_type == 'ALSTM':
            v = new_velocity
            s = new_scale

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            if self.rnn_type == 'MLSTM' or self.rnn_type == 'NLSTM':
                return result, hidden, v, raw_outputs, outputs
            elif self.rnn_type == 'ALSTM':
                return result, hidden, v, s, raw_outputs, outputs
            else:
                return result, hidden, raw_outputs, outputs
        if self.rnn_type == 'MLSTM' or self.rnn_type == 'NLSTM':
            return result, hidden, v
        elif self.rnn_type == 'ALSTM':
            return result, hidden, v, s
        else:
            return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'MLSTM' or self.rnn_type == 'NLSTM':
            hidden = [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
            v = [weight.new(1, bsz, 4*self.nhid if l != self.nlayers - 1 else (4*self.ninp if self.tie_weights else 4*self.nhid)).zero_()
                    for l in range(self.nlayers)]
            return hidden, v
        elif self.rnn_type == 'ALSTM':
            hidden = [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
            v = [weight.new(1, bsz, 4*self.nhid if l != self.nlayers - 1 else (4*self.ninp if self.tie_weights else 4*self.nhid)).zero_()
                    for l in range(self.nlayers)]
            s = [weight.new(1, bsz, 4*self.nhid if l != self.nlayers - 1 else (4*self.ninp if self.tie_weights else 4*self.nhid)).zero_()
                    for l in range(self.nlayers)]
            return hidden, v, s
        elif self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
