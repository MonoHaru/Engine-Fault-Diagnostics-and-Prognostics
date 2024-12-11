
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import warnings
from function_file.ML_functions import *
from function_file.time_series import time_series_dataframe
import time
from tqdm.notebook import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings('ignore')


################################################ Transformer ################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout = 0.5):
        super(PositionalEncoding, self).__init__()   
        self.dropout = nn.Dropout(p = dropout)    
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.5):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, dropout = dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

###########################################################################################################################


def new_multistep_time_series(temp_data, label_data, input_window, output_window):
    inout_seq = []
    label = []
    batch_len = input_window + output_window
    L = len(temp_data)
    for i in range(L-batch_len):
        train_seq = temp_data[i : i + input_window]
        train_label = temp_data[i + output_window : i + output_window + input_window] #[40 : ]
        min_temp_label = min(label_data[i : i+output_window+input_window])
        max_temp_label = max(label_data[i : i+output_window+input_window])
        
        if min_temp_label == max_temp_label:
            inout_seq.append((train_seq, train_label))
            label.append(max_temp_label)
        else:
            continue

    return torch.FloatTensor(inout_seq), label