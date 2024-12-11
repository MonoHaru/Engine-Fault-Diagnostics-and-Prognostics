
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

#Transformer
################################################################################
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
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
################################################################################

# Dataset
class TimeSeiresDataset(Dataset):
    def __init__(self, X,y, input_window):
        self.input_window = input_window
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx].type(torch.float32), self.y[idx].type(torch.float32)
    
    def __len__(self):
        return len(self.X)

# Data Preprocessing
def multistep_time_series(temp_data, label_data, input_window, output_window):
    inout_seq = []
    label = []
    batch_len= input_window + output_window
    L = len(temp_data)
    for i in range(L-batch_len):
        train_seq = temp_data[i:i+input_window]
        train_label = temp_data[i+output_window:i+input_window+output_window]
        temp_label = max(label_data[i+output_window:i+input_window+output_window])
        
        inout_seq.append((train_seq ,train_label))
        label.append(temp_label)
    return torch.FloatTensor(inout_seq), label


############################################################################
# def create_inout_sequences(input_data, tw, output_window):
#     inout_seq = []
#     L = len(input_data)
#     for i in range(L-tw):
#         train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
#         train_label = input_data[i:i+tw]
#         #train_label = input_data[i+output_window:i+tw+output_window]
#         inout_seq.append((train_seq ,train_label))
#     return torch.FloatTensor(inout_seq)


# def get_batch(source, i,batch_size):
#     seq_len = min(batch_size, len(source) - 1 - i)
#     data = source[i:i+seq_len]    
#     input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
#     target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
#     return input, target

# def get_data():
#     #amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
#     #amplitude = scaler.fit_transform(df.reshape(-1, 1)).reshape(-1)
#     #from pandas import read_csv
#     #series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

#     df = time_series_dataframe()
#     df = df['TEMP'].values   
    
#     train_len = int(len(df) * 0.7)
#     train_data = df[:train_len] 
#     test_data = df[train_len:] 
#     train_data = scaler_train.fit_transform(train_data.reshape(-1,1)).reshape(-1)
#     test_data = scaler_test.fit_transform(test_data.reshape(-1,1)).reshape(-1)

#     # convert our train data into a pytorch train tensor
#     #train_tensor = torch.FloatTensor(train_data).view(-1)
#     # todo: add comment.. 
#     train_sequence = create_inout_sequences(train_data,input_window)
#     #train_sequence = train_sequence[:-output_window] #todo: fix hack?

#     #test_data = torch.FloatTensor(test_data).view(-1) 
#     test_data = create_inout_sequences(test_data,input_window)
#     #test_data = test_data[:-output_window] #todo: fix hack?

#     return train_sequence.to(device), test_data.to(device)
############################################################################

def get_batch(source, i,batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target


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
            
############################################################################


#Train
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

# def train2():
#     model.train() # Turn on the train mode
#     total_loss = 0.
#     start_time = time.time()

#     for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
#         data, targets = get_batch(train_data, i,batch_size)
#         optimizer.zero_grad()
#         output = model(data)        

#         if calculate_loss_over_all_values:
#             loss = criterion(output, targets)
#         else:
#             loss = criterion(output[-output_window:], targets[-output_window:])
    
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()

#         total_loss += loss.item()
#         log_interval = int(len(train_data) / batch_size / 5)
#         if batch % log_interval == 0 and batch > 0:
#             cur_loss = total_loss / log_interval
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches | '
#                   'lr {:02.6f} | {:5.2f} ms | '
#                   'loss {:5.5f} | ppl {:8.2f}'.format(
#                     epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
#                     elapsed * 1000 / log_interval,
#                     cur_loss, math.exp(cur_loss)))
#             total_loss = 0
#             start_time = time.time()


# validation
def calculate_loss_and_plot(model, test_dataloader, device, criterion, output_window, scaler_test,batch_size):
    model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    result_to_ML = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader)):
            input_batch, label_batch = batch[0].to(device), batch[1].to(device)
            
            for jdx in range(input_batch.shape[0]):
                input = input_batch[jdx].unsqueeze(0)
                label = label_batch[jdx].unsqueeze(0)
                output = model(input)
                loss = criterion(output, label)
                total_loss += loss.item()
                test_result = torch.cat((test_result, output[:, -output_window:].view(-1).cpu()), 0) #todo: check this. -> looks good to me
                truth = torch.cat((truth, label[:, -output_window:].view(-1).cpu()), 0)
                result_to_ML.append(output[:, -output_window:].view(-1).cpu().detach().numpy())
    
    test_result = scaler_test.inverse_transform(test_result.reshape(-1,1)).reshape(-1)
    truth = scaler_test.inverse_transform(truth.reshape(-1,1)).reshape(-1)
    
    plt.plot(test_result, label = 'prediction')
    plt.plot(truth, label = 'truth')
    plt.grid(True, which = 'both')
    plt.ylim([500,1000])
    plt.axhline(y=0, color='k')
    plt.show()
    plt.close()
    

    return truth, test_result, result_to_ML, total_loss / idx

calculate_loss_over_all_values =  False

def plot_and_loss(model, data_source, criterion,input_window, output_window, scaler_DL):
    model.eval() 
    print("Evaluation...")
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
    #pyplot.plot(test_result-truth,color="green")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    plt.close()
    
    return truth, test_result, result_to_ML, total_loss / i


def evaluate(model, test_dataloader, device, criterion, output_window):
    model.eval()
    total_loss = 0.0
    batch_size = 512
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input, label = batch[0].to(device), batch[1].to(device)
            output = model(input)
            total_loss += criterion(output[:, -output_window:], label[:, -output_window:]).item()

    return total_loss / (len(test_dataloader) * batch_size)

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
            
    