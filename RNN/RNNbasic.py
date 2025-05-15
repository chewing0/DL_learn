import torch
from matplotlib import pyplot as plt
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, TensorDataset

def generate_data_from_sin(mean: float, std: float, size: int, tau: int) -> tuple[Tensor, Tensor]:
    time_steps = torch.arange(start=1, end=size + tau + 1, dtype=torch.float)
    values = torch.sin(time_steps * 0.01) + torch.normal(mean, std, size=(size + tau,))

    return time_steps, values


def get_features_and_labels(values: Tensor, size: int, tau: int) -> tuple[Tensor, Tensor]:
    features = torch.stack([values[i:i + tau] for i in range(size)], dim=0)
    labels = values[tau:]

    return features, labels

def get_dataloader(feature, labels, train_size, batch_size):
    train_dataset = TensorDataset(feature[:train_size], labels[:train_size])
    valid_dataset = TensorDataset(feature[train_size:], labels[train_size:])
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_iter = DataLoader(valid_dataset, batch_size, shuffle=False)
    return train_iter, valid_iter

sample_size = 1000
tau = 4
epochs = 5
lr = 0.01
batch_size = 16

# 定义网络
net = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10,1)
)
loss_fun = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

# 数据生成
time_steps, value = generate_data_from_sin(mean=0, std=0.2, size=sample_size, tau=tau)
features, labels = get_features_and_labels(values=value, size=sample_size, tau=tau)
train_iter, test_iter = get_dataloader(feature=features, labels=labels, train_size=int(0.8*sample_size), batch_size=batch_size)

for epoch in range(epochs):
    net.train()
    train_loss_accu = 0
    for feature, label in train_iter:
        optimizer.zero_grad()
        pre = net(feature)
        loss = loss_fun(pre.squeeze(), label)
        loss.backward()
        optimizer.step()
        train_loss_accu += loss.item()

    net.eval()
    valid_loss_accu = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_iter:
            predictions = net(batch_features)
            loss = loss_fun(predictions.squeeze(), batch_labels)
            valid_loss_accu += loss.item()

    train_loss = train_loss_accu / len(test_iter)
    valid_loss = valid_loss_accu / len(test_iter)
    print(f"第 {epoch + 1}/{epochs} 轮，训练损失：{train_loss :.4f}，测试损失：{valid_loss :.4f}")
