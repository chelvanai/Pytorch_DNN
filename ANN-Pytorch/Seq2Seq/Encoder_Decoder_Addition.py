import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Addition dataset create function
def create_addition_dataset(n_samples=1000, max_val=10):
    x = torch.randint(0, max_val, (n_samples, 2))
    y = x[:, 0] + x[:, 1]
    return x, y

# make dataset
x, y = create_addition_dataset()

# Encoder 
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

# Decoder 
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Encoder-Decoder
class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Encoder_Decoder(2, 10, 1)

criterian = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training
def train(model, x, y, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterian(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss.item():.4f}')



X = torch.from_numpy(x.numpy()).float()
Y = torch.from_numpy(y.numpy()).float()

train(model, X, Y)