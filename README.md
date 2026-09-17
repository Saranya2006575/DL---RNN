# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

The objective is to predict future stock prices based on historical closing prices. The historical stock price dataset is loaded from a CSV file, and the Close price is used as the input feature. The data is normalized and converted into sequences before being given to the RNN model.

The dataset is divided into training and testing data. The RNN model is trained using historical stock prices and then used to predict stock prices for the test data. The actual and predicted prices are plotted for comparison.

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 
Convert data to tensors and set up DataLoader.  

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 
Train the model with loss tracking.


### STEP 6: 

Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: V.SHREYA

### Register Number:212224230266

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv("trainset.csv")
df_test = pd.read_csv("testset.csv")

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
train_prices = scaler.fit_transform(train_prices)
test_prices = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(train_prices, seq_length)
x_test, y_test = create_sequences(test_prices, seq_length)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
## Step 3: Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")
# Plot training loss

plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()
    
# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="852" height="675" alt="image" src="https://github.com/user-attachments/assets/984fe271-9088-45a3-a838-d712a07106ab" />


## True Stock Price, Predicted Stock Price vs time

<img width="1283" height="792" alt="image" src="https://github.com/user-attachments/assets/27bb67eb-c4e9-4226-bbf1-f8cae740222e" />

### Predictions
<img width="366" height="106" alt="image" src="https://github.com/user-attachments/assets/44244669-c1e4-42b5-83a8-6551d69a203f" />


## RESULT
The Recurrent Neural Network (RNN) model was successfully developed and trained using historical stock closing price data. The stock prices were normalized and converted into time-series sequences before training. The trained model successfully predicted stock prices on the test dataset, and the actual and predicted prices were visualized using graphs. Thus, an RNN model was successfully implemented for stock price prediction.
