"""Verbatim DGCRN model source.

Vendored from CauAir (src/models/dgcrn.py).
BaseModel replaced with nn.Module; explicit params stored on self.
Reference: https://github.com/tsinghua-fib-lab/Traffic-Benchmark/tree/master/methods/DGCRN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(gcn, self).__init__()
        if type == 'RNN':
            self.gconv = gconv_RNN()
            self.gconv_preA = gconv_hyper()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])
        elif type == 'hyper':
            self.gconv = gconv_hyper()
            self.mlp = nn.Sequential(
                OrderedDict([('fc1', nn.Linear((gdep + 1) * dims[0], dims[1])),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(dims[1], dims[2])),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(dims[2], dims[3]))]))
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

    def forward(self, x, adj):
        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            for _ in range(self.gdep):
                h = self.alpha * x + self.beta * self.gconv(
                    h, adj[0]) + self.gamma * self.gconv_preA(h, adj[1])
                out.append(h)
        else:
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho


class DGCRN(nn.Module):
    """Dynamic Graph Convolutional Recurrent Network."""

    def __init__(self, device, node_num, input_dim, output_dim, seq_len, horizon,
                 predefined_adj=None, gcn_depth=2, rnn_size=64, hyperGNN_dim=16,
                 node_dim=40, middle_dim=2, list_weight=None, tpd=288,
                 tanhalpha=3, cl_decay_step=2000, dropout=0.3):
        super(DGCRN, self).__init__()
        self.device = device
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.predefined_adj = predefined_adj
        self.hidden_size = rnn_size
        self.tpd = tpd
        self.alpha = tanhalpha
        self.cl_decay_step = cl_decay_step
        self.use_curriculum_learning = False

        if list_weight is None:
            list_weight = [0.05, 0.95, 0.95]

        self.emb1 = nn.Embedding(node_num, node_dim)
        self.emb2 = nn.Embedding(node_num, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(node_num).to(device)

        dims_hyper = [self.hidden_size + self.input_dim, hyperGNN_dim, middle_dim, node_dim]
        self.GCN1_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN1_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN1_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN1_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        dims = [self.input_dim + self.hidden_size, self.hidden_size]
        self.gz1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')

    def preprocessing(self, adj, predefined_adj):
        adj = adj + torch.eye(self.node_num).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_adj]

    def step(self, input, Hidden_State, Cell_State, predefined_adj, type='encoder', i=None):
        x = input
        x = x.transpose(1, 2).contiguous()
        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)
        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self.node_num, self.hidden_size)), 2)

        if type == 'encoder':
            filter1 = self.GCN1_tg(hyper_input, predefined_adj[0]) + \
                      self.GCN1_tg_1(hyper_input, predefined_adj[1])
            filter2 = self.GCN2_tg(hyper_input, predefined_adj[0]) + \
                      self.GCN2_tg_1(hyper_input, predefined_adj[1])
        if type == 'decoder':
            filter1 = self.GCN1_tg_de(hyper_input, predefined_adj[0]) + \
                      self.GCN1_tg_de_1(hyper_input, predefined_adj[1])
            filter2 = self.GCN2_tg_de(hyper_input, predefined_adj[0]) + \
                      self.GCN2_tg_de_1(hyper_input, predefined_adj[1])

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))
        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1))
        adj = F.relu(torch.tanh(self.alpha * a))
        adp = self.preprocessing(adj, predefined_adj[0])
        adpT = self.preprocessing(adj.transpose(1, 2), predefined_adj[1])

        Hidden_State = Hidden_State.view(-1, self.node_num, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.node_num, self.hidden_size)
        combined = torch.cat((x, Hidden_State), -1)

        if type == 'encoder':
            z = torch.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
            r = torch.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))
            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))
        elif type == 'decoder':
            z = torch.sigmoid(self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = torch.sigmoid(self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))
            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul(1 - z, Cell_State)
        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(-1, self.hidden_size)

    def initHidden(self, batch_size, hidden_size):
        Hidden_State = torch.zeros(batch_size, hidden_size).to(self.device)
        Cell_State = torch.zeros(batch_size, hidden_size).to(self.device)
        return Hidden_State, Cell_State

    def forward(self, input, label=None):
        """Forward pass. input: (B, T, N, F) -> (B, horizon, N, output_dim)"""
        x = input.transpose(1, 3)  # (B, F, N, T)
        batch_size = x.size(0)
        Hidden_State, Cell_State = self.initHidden(
            batch_size * self.node_num, self.hidden_size)

        for i in range(self.seq_len):
            Hidden_State, Cell_State = self.step(
                torch.squeeze(x[..., i]), Hidden_State, Cell_State,
                self.predefined_adj, 'encoder', i)

        # Decoder: auto-regressive
        # Decoder input needs input_dim channels (output_dim + covariates)
        decoder_input = torch.zeros(
            (batch_size, self.input_dim, self.node_num), device=self.device)
        outputs_final = []
        for i in range(self.horizon):
            Hidden_State, Cell_State = self.step(
                decoder_input, Hidden_State, Cell_State,
                self.predefined_adj, 'decoder', None)
            decoder_output = self.fc_final(Hidden_State)
            decoder_input_val = decoder_output.view(
                batch_size, self.node_num, self.output_dim).transpose(1, 2)
            # Pad with zeros for covariate channels
            pad = torch.zeros(
                batch_size, self.input_dim - self.output_dim, self.node_num,
                device=self.device)
            decoder_input = torch.cat([decoder_input_val, pad], dim=1)
            outputs_final.append(decoder_output)

        outputs_final = torch.stack(outputs_final, dim=1)
        outputs_final = outputs_final.view(
            batch_size, self.node_num, self.horizon, self.output_dim).transpose(1, 2)
        return outputs_final
