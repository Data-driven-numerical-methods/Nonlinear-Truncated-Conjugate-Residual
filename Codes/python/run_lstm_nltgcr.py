# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

training_set = pd.read_csv('airline-passengers.csv')
training_set = pd.read_csv('shampoo.csv')

training_set = training_set.iloc[:,1:2].values

plt.plot(training_set, label = 'Shampoo Sales Data')
plt.show()

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

seq_length = 3
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        return out
    
num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1

model = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression


def FF(x, y):
    # reload(w)
    y_pred = model(x)
    f = criterion(y_pred, y)
    f.backward()
    vl = []
    for param in model.parameters():
        if param.grad is not None:
            vv = nn.functional.normalize(param.grad, p=2.0, dim = 0)
            v = vv.view(-1)
            vl.append(v)
            fp = torch.cat(vl)  
    model.zero_grad()
    return fp

def reload(fp):
    offset = 0
    for name, param in model.named_parameters():
        shape = param.shape
        param.data = fp[offset: offset + shape.numel()].view(shape)
        offset = offset + shape.numel()
        
def combine(model):
    vl = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            v = param.data.view(-1)
            vl.append(v)
    fp = torch.cat(vl)    
    return fp

G_dict = {}
for name in model.named_parameters():
    G_dict[name[0]] = name[1].data
  
sizeG = {}
for key in G_dict:
    sizeG[key] = G_dict[key].shape
w =  combine(model)
sum_G = sum(v.numel() for _, v in G_dict.items())
assert(len(w) ==sum_G)
d = len(w)
lb= 1
device = 'cpu'
P = torch.zeros((d, lb), requires_grad=False).to(device)
AP = torch.zeros((d,lb), requires_grad=False).to(device)
reload(w)
r = FF(trainX.to(device), trainY.to(device))
rho = torch.norm(r)
epsf = 1e-1
ep = epsf * rho/torch.norm(w)
w1 = w-ep*r
reload(w1)
Ar = (FF(trainX.to(device), trainY.to(device))-r)/ep
reload(w)
t = torch.norm(Ar)
t = 1.0/t
P[:,0] = t * r
AP[:,0]=  t * Ar 
loss_list = []
i2 = 1
i = 1
for epoch in range(num_epochs):
    correct = 0
    model.train()

    
    alph = AP.t()@r
    with torch.no_grad():
        dire = P@alph
        w = w + dire
        reload(w)
    r = FF(trainX, trainY)
    with torch.no_grad():
        rho = torch.norm(r)
        w1 = w-ep*r
        reload(w1)
    r1 = FF(trainX, trainY)
    Ar = (r1-r)/ep
    reload(w)
    ep = epsf * rho/torch.norm(w)
    p = r
    if i <= lb:
        k = 0
    else:
        k = i2
    while True:
        if k ==lb:
            k = 0
        k +=1
        tau = torch.inner(Ar, AP[:,k-1])
        p = p - tau*(P[:,k-1])
        Ar = Ar -  tau*(AP[:,k-1])
        if k == i2:
            break
    t = torch.norm(Ar)
    if (i2) == lb:
        i2 = 0
    i2 = i2+1
    i = i+1
    t = 1.0/t
    AP[:,i2-1] = t*Ar
    P[:,i2-1] = t*p
    if epoch % 100 == 0:
        with torch.no_grad():
            outputs = model(testX)
            loss = criterion(outputs, testY)
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        loss_list.append(loss.item())
        
with open("results/loss_nltgcrs.pkl", "wb") as fp:  
    pickle.dump(loss_list, fp)
model.eval()
train_predict = model(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()

model.eval()
train_predict = model(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()