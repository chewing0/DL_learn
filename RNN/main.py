import torch
from torch import nn
from torch.nn import functional as F

from dataloader import load_data
from train import train_ch8
from net import RNNModel

if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*30)
    print(f"Using device: {device}")
    print("="*30)

    batch_size = 32
    num_steps = 35
    num_hiddens = 256
    num_epochs = 500
    lr = 1.0

    file_path = r'learning\data\timemachine.txt'
    # 加载数据
    train_iter, vocab  = load_data(batch_size, num_steps, file_path=file_path)

    num_inputs = len(vocab)
    # 定义RNN网络
    # rnn_layer = nn.RNN(num_inputs, num_hiddens)
    # gru_layer = nn.GRU(num_inputs, num_hiddens)
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, 2)
    net = RNNModel(lstm_layer, vocab_size=len(vocab))

    # 将模型移动到设备
    net = net.to(device)

    # 训练模型
    # 输入参数：网络，训练数据加载器，词汇表，学习率，训练轮数，设备
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)