# -*- coding: utf-8 -*-
# file: asgcn.py
# author:  <gene_zhangchen@163.com>
# Copyright (C) 2020. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    # def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
    #     batch_size = x.shape[0]
    #     seq_len = x.shape[1]
    #     aspect_double_idx = aspect_double_idx.cpu().numpy()
    #     text_len = text_len.cpu().numpy()
    #     aspect_len = aspect_len.cpu().numpy()
    #     weight = [[] for i in range(batch_size)]
    #     for i in range(batch_size):
    #         context_len = text_len[i] - aspect_len[i]
    #         for j in range(aspect_double_idx[i,0]):
    #             weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
    #         for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
    #             weight[i].append(0)
    #         for j in range(aspect_double_idx[i,1]+1, text_len[i]):
    #             weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
    #         for j in range(text_len[i], seq_len):
    #             weight[i].append(0)
    #     weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
    #     return weight*x
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        
        weight = []
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            current_weight = []
            
            for j in range(aspect_double_idx[i, 0]):
                current_weight.append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                current_weight.append(0)
            
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                current_weight.append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            
            # Fill remaining sequence length with zeros if necessary
            while len(current_weight) < seq_len:
                current_weight.append(0)
            
            # Trim if the length is too long
            current_weight = current_weight[:seq_len]
            
            weight.append(current_weight)
        
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight * x

    # def mask(self, x, aspect_double_idx):
    #     batch_size, seq_len = x.shape[0], x.shape[1]
    #     aspect_double_idx = aspect_double_idx.cpu().numpy()
    #     mask = [[] for i in range(batch_size)]
    #     for i in range(batch_size):
    #         for j in range(aspect_double_idx[i,0]):
    #             mask[i].append(0)
    #         for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
    #             mask[i].append(1)
    #         for j in range(aspect_double_idx[i,1]+1, seq_len):
    #             mask[i].append(0)
    #     mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
    #     return mask*x
    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        
        mask = []
        for i in range(batch_size):
            current_mask = []
            
            # Add 0s before the aspect
            for j in range(aspect_double_idx[i, 0]):
                current_mask.append(0)
            
            # Add 1s for the aspect span
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                current_mask.append(1)
            
            # Add 0s after the aspect
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                current_mask.append(0)
            
            # Adjust the length dynamically
            # If the current mask is shorter, pad with 0s
            while len(current_mask) < seq_len:
                current_mask.append(0)
            
            # If it's longer, truncate to the correct length
            current_mask = current_mask[:seq_len]
            
            mask.append(current_mask)
        
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask * x
    

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output