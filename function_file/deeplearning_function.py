from copy import deepcopy as dc
import numpy as np
import torch 
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    
    for i in range(1,n_steps+1):
        df[f'TEMP(t-{i})'] = df['TEMP'].shift(i)

    df.dropna(inplace=True)
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
def DEEPLEARNING(datafrmae, lookback, batch_size, front_or_back):
    shifted_df = prepare_dataframe_for_lstm(datafrmae, lookback)
    shifted_df.drop(columns='TIME', inplace = True)

    X = shifted_df.values
    X = scaler.fit_transform(X)
    y = X[:, 0]
    X = X[:, 1:]

    if front_or_back == 'back':
        train_len = int(len(X) * 0.8)
        X_train = X[:train_len]
        X_test = X[train_len:]
        y_train = y[:train_len]
        y_test = y[train_len:]


    else:
        train_len = int(len(X) * 0.1)
        X_train = X[train_len:]
        X_test = X[:train_len]
        y_train = y[train_len:]
        y_test = y[:train_len]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1,1))
    y_test = y_test.reshape((-1,1))

    # Tranform to Torch Tensor
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, test_loader, X_test, y_test


def plot_prediction(model, X_test, y_test, lookback):
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y_test = dc(dummies[:, 0])
    
    return test_predictions, new_y_test