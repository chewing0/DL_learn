import torch
from torch import nn
from torch.nn import functional as F

from dataloader import load_data
from train import train_ch8
from net import RNNModel

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
num_steps = 35

# 加载数据
train_iter, vocab  = load_data(batch_size, num_steps)

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, device)