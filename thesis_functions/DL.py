import os
import numpy as np
import pandas as pd

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

calculate_loss_over_all_values = False


def plot_and_loss2(model, data_source, criterion,input_window, output_window, scaler_DL):
    model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    result_to_ML = []
    with torch.no_grad():
        for i in tqdm(range(len(data_source)-1)):
            data, target = get_batch(data_source, i,1, input_window)
            # look like the model returns static values for the output window
            output = model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            result_to_ML.append(output[-output_window:].view(-1).cpu().detach().numpy())
            
    test_result = scaler_DL.inverse_transform(test_result.reshape(-1,1)).reshape(-1)
    truth = scaler_DL.inverse_transform(truth.reshape(-1,1)).reshape(-1)
    
    plt.plot(test_result,label = 'Prediction')
    plt.plot(truth,label = 'Truth')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    plt.close()
    
    return truth, test_result, result_to_ML, total_loss / i

def get_batch(source, i,batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target

def evaluate2(model, data_source, criterion, output_window, input_window):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 256
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size, input_window)
            output = model(data)            
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
    return total_loss / len(data_source)
            

def train(model, train_dataloader, device, optimizer, criterion, epoch, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.0
    
    for idx, batch in enumerate(train_dataloader):
        input, label = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_dataloader)  / 5)
        
        if idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('|epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | loss {:5.5f}'.format(epoch, idx, len(train_dataloader),
                                                                                           scheduler.get_lr()[0], cur_loss ))
            total_loss = 0
            start_time = time.time()
            
def train_tmp(model, train_data,batch_size, optimizer, criterion, input_window, output_window, epoch, scheduler):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size, input_window)
        optimizer.zero_grad()
        output = model(data)        

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f} |'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            
            
def new_multistep_time_series(temp_data, label_data, input_window, output_window):
    inout_seq = []
    label = []
    batch_len = input_window + output_window
    L = len(temp_data)
    for i in range(L-batch_len):
        train_seq = temp_data[i : i + input_window]
        train_label = temp_data[i + output_window : i + output_window + input_window] #[40 : ]
        min_temp_label = min(label_data[i + output_window : i+output_window+input_window])
        max_temp_label = max(label_data[i + output_window : i+output_window+input_window])
        
        if min_temp_label == max_temp_label:
            inout_seq.append((train_seq, train_label))
            label.append(max_temp_label)
        else:
            continue
            
    return torch.FloatTensor(inout_seq), label

###########################################################################################################################


def trend(time, slope = 0):
    return time * slope

def time_series_dataframe_ML():
    path_temp_gps = './temp_add_gps/'
    list_temp_gps = os.listdir(path_temp_gps)

    m1 = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[0]))
    for i in range(1,8):
        tmp = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[i]))
        m1 = pd.concat([m1, tmp], axis = 0)

    m1 = m1.reset_index(drop = True)
    m1 = m1[m1['TEMP']>=243.07].reset_index(drop = True)
    time_df = m1
    time_df = time_df.loc[:, ['TEMP']]

    for i in range(1,8):
        globals()['df_'+str(i)+'_temp'] = time_df[60436*(i-1):60436*i].reset_index(drop = True)

    N = 6
    dx = (600 - df_1_temp['TEMP'].mean()) / N # ??? ??????? ???? ?????? : 56.3785
    dx_minute = dx / (len(df_1_temp)-1) # ???? ??????

    time = np.arange(len(df_1_temp))
    slope = dx_minute * 2



    for i in range(1,8):
        mean = globals()['df_'+str(i)+'_temp']['TEMP'].mean()
        diff  = 280.40784269309677 - mean
        globals()['df_'+str(i)+'_temp']['TEMP'] += diff

    for i in range(8,13): # 8, 9, 10, 11, 12
        globals()['df_'+str(i)+'_temp'] = globals()['df_'+str(i-7)+'_temp'].copy()

    for i in range(2,13):
        series = np.round(trend(time, slope = slope) + globals()['df_'+str(i)+'_temp']['TEMP'] + dx*(i-2), 3)
        globals()['df_'+str(i)+'_temp']['TEMP'] = series

    temp_len = len(df_1_temp) * 12
    label = np.array([[i] * 120872 for i in range(1,7)]).reshape(-1)
    temp_TIME = pd.DataFrame({'TIME' : np.arange(temp_len)})
    temp_label = pd.DataFrame({'label' : label})
    df_temp_all = pd.concat([df_1_temp, df_2_temp, df_3_temp, df_4_temp, df_5_temp, df_6_temp, df_7_temp, df_8_temp, df_9_temp, df_10_temp, df_11_temp, df_12_temp], axis = 0)
    df_temp_all = df_temp_all.reset_index(drop = True)
    df_temp_all = pd.concat([df_temp_all,temp_label, temp_TIME], axis = 1)
    
    return df_temp_all



