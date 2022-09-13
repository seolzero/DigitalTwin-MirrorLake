import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.preprocessing import MinMaxScaler
import time
import datetime
import os

data_file = ""
LOOKBACK_SIZE = 10
for dirname, _, filenames in os.walk('c:/Users/seol/PycharmProjects/anomaly-detection-deepant/csv'):
    for filename in filenames:
        data_file = os.path.join(dirname, filename)


def read_modulate_data(data_file):
    """
        Data ingestion : Function to read and formulate the data
    """

    data = pd.read_csv(data_file)
    data.fillna(data.mean(numeric_only=True), inplace=True)
    df = data.copy()
    data.set_index("timestamp", inplace=True)
    data.index = pd.to_datetime(data.index)
    return data, df


def data_pre_processing(df):
    """
        Data pre-processing : Function to create data for Model
    """
    try:
        scaled_data = MinMaxScaler(feature_range=(0, 1))
        data_scaled_ = scaled_data.fit_transform(df)
        df.loc[:, :] = data_scaled_
        _data_ = df.to_numpy(copy=True)
        X = np.zeros(shape=(df.shape[0] - LOOKBACK_SIZE, LOOKBACK_SIZE, df.shape[1]))
        Y = np.zeros(shape=(df.shape[0] - LOOKBACK_SIZE, df.shape[1]))
        timesteps = []
        for i in range(LOOKBACK_SIZE - 1, df.shape[0] - 1):
            timesteps.append(df.index[i])
            Y[i - LOOKBACK_SIZE + 1] = _data_[i + 1]
            for j in range(i - LOOKBACK_SIZE + 1, i + 1):
                X[i - LOOKBACK_SIZE + 1][LOOKBACK_SIZE - 1 - i + j] = _data_[j]

        return X, Y, timesteps
    except Exception as e:
        print("Error while performing data pre-processing : {0}".format(e))
        return None, None, None


class DeepAnT(torch.nn.Module):
    """
        Model : Class for DeepAnT model
    """

    def __init__(self, DIMENSION):
        super(DeepAnT, self).__init__()
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=DIMENSION, out_channels=16, kernel_size=3)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(16, 40)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.dense_2_layer = torch.nn.Linear(40, DIMENSION)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d_1_layer(x)
        x = self.relu_1_layer(x)
        x = self.maxpooling_1_layer(x)
        x = self.conv1d_2_layer(x)
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_1_layer(x)
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        return self.dense_2_layer(x)


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step


def compute(X, Y):
    """
        Computation : Find Anomaly using model based computation
    """
    model = DeepAnT(26)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)
    train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.float32)))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    train_step = make_train_step(model, criterion, optimizer)
    for epoch in range(5):
        loss_sum = 0.0
        ctr = 0
        for x_batch, y_batch in train_loader:
            loss_train = train_step(x_batch, y_batch)
            loss_sum += loss_train
            ctr += 1
        print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum / ctr), epoch + 1))

    torch.save(model, f'./saveCrainDeepAnTmodel.pt')
    print('model save')
    print(torch.tensor(X.astype(np.float32)).shape)

    hypothesis = model(torch.tensor(X.astype(np.float32))).detach().numpy()
    loss = np.linalg.norm(hypothesis - Y, axis=1)
    # print(loss.shape)
    return loss.reshape(len(loss), 1)

def main():
    count = 0
    data, _data = read_modulate_data(data_file)
    X, Y, T = data_pre_processing(data)
    loss = compute(X, Y)
    print('=loss========================')
    print(len(loss))
    for index, value in enumerate(loss):
        print(index, value)
        if(value > 1.8):
            print(index, value)
            count+=1
    print('=count========================')
    print(count)


if __name__ == '__main__':
    main()