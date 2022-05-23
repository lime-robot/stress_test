import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
#import fast_former


### Model
class SognaModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        input_dim = len(args.cont_cols) + 3 + 3
                
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 2, padding='same',
                      padding_mode='replicate'),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 3, padding='same',
                      padding_mode='replicate'),
            nn.Mish(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 4, padding='same',
                      padding_mode='replicate'),
            nn.Mish(),
        )

        self.lstm1 = nn.LSTM(input_dim * 4, 1024, batch_first=True,
                             bidirectional=True, dropout=args.dropout)
        self.lstm2 = nn.LSTM(1024 * 2, 512, batch_first=True,
                             bidirectional=True, dropout=args.dropout)
        self.lstm3 = nn.LSTM(512 * 2, 256, batch_first=True, bidirectional=True,
                             dropout=args.dropout)
        self.lstm4 = nn.LSTM(256 * 2, 128, batch_first=True, bidirectional=True,
                             dropout=args.dropout)

        self.reg_layer = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.SELU(),
            nn.Linear(128, 5),
        )

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

            if 'conv' in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

            elif 'reg' in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

            elif 'emb' in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(p.data)

    def forward(self, batch):
        x = torch.cat([batch['features'],
                       batch['r_code'], batch['c_code']], dim=-1)

        x = x.permute(0, 2, 1)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        x_conv3 = self.conv3(x)

        x = torch.cat([x, x_conv1, x_conv2, x_conv3], dim=1)

        x = x.permute(0, 2, 1)

        features, _ = self.lstm1(x)
        features, _ = self.lstm2(features)
        features, _ = self.lstm3(features)
        features, _ = self.lstm4(features)

        pred = self.reg_layer(features)
        
        res = {}
        res['prediction'] = pred[:, :, 0]
        
        if 'target' in batch:            
            loss = F.l1_loss(pred, batch['target'], reduction='none')
            res['loss'] = loss[batch['u_out']==0].mean()
            with torch.no_grad():
                res['mae'] = loss[:, :, 0][batch['u_out']==0].mean()
        return res




