# 01线性回归
import torch
from torch.utils import data
from torch import nn


# 数据生成
# .unsqueeze(1)的作用是什么
def data_maker(weights: torch.Tensor, bias: float, size:int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.normal(mean=0, std=1, size=(size, len(weights)))
    y = x @ weights + bias
    # 引入随机的噪声
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return x, y.unsqueeze(1)

if __name__ == '__main__':
    ture_w = torch.tensor([2.3, 1.4])
    ture_b = 1.3
    features, labels = data_maker(ture_w, ture_b, 1000)

    # 数据处理和批量化数据集
    batch_size = 32
    dataset = data.TensorDataset(features, labels)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

    # 初始化网络
    net = nn.Sequential(nn.Linear(in_features=2, out_features=1))
    loss = nn.MSELoss()


    # 初始化参数
    nn.init.normal_(net[0].weight, mean=0, std=0.01)
    nn.init.zeros_(net[0].bias)

    # 定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    epochs = 3
    for epoch in range(epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch = {epoch + 1}, loss = {l:f}')

    for name, param in net.named_parameters():
        print(f'{name} = {param.data.squeeze().tolist()}')