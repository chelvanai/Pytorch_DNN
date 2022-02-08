import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


dataset = np.array(range(1,10))
dataset = np.expand_dims(dataset, axis=1)

transformer = MinMaxScaler(feature_range=(0,1))
dataset = transformer.fit_transform(dataset)


# split window
window_size = 3
def split_window(dataset, window_size):
    dataX = []
    dataY = []
    for i in range(len(dataset)-window_size):
        dataX.append(dataset[i:i+window_size])
        dataY.append(dataset[i+window_size])
    return np.array(dataX), np.array(dataY)

x,y = split_window(dataset, window_size)


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[-1])
        return out

model = LSTM(1,10,1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)

# train
epochs = 100
losses = []
for epoch in range(epochs):
    inputs = Variable(torch.from_numpy(x)).float()
    targets = Variable(torch.from_numpy(y)).float()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.data[0])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch {}/{}'.format(epoch, epochs), 'Loss: {:.4f}'.format(loss.data[0]))

# predict
predicted = model(Variable(torch.from_numpy(x[-window_size:])).float()).data.numpy()
res = transformer.inverse_transform(predicted)
print(res)

# Next 10 days prediction
l = []
for i in range(10):
    l.append(res[-1])
    x = np.append(x[-window_size:], l)
    predicted = model(Variable(torch.from_numpy(x[-window_size:])).float()).data.numpy()
    res = transformer.inverse_transform(predicted)
    print(res)
    
