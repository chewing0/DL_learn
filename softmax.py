# 02 softmax模型实现
import torch
import torchvision
from torch.utils import data
from torchvision import datasets
from torch import nn

def train_one_epoch(net, train_data, optimizer, loss_fun):
    net.train()
    total_loss = 0
    for imgs, labels in train_data:
        optimizer.zero_grad()
        output = net(imgs)
        loss_value = loss_fun(output, labels)
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()
    avg_loss = total_loss/len(train_data)
    return avg_loss

def validate(net, test_data):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_data:
            output = net(imgs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def init_nn(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if __name__ == '__main__':
    # 参数定义
    batch_size = 256

    # 数据集处理
    train_datasets = datasets.MNIST(root='./DeepLearning/data',
                                train=True,
                                download=True,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                    ))
    test_dataset = datasets.MNIST(root='./DeepLearning/data',
                                train=False,
                                download=True,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                    ))
    train_iter = data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # 定义网络
    net = nn.Sequential(nn.Flatten(), nn.Linear(in_features=28*28, out_features=10))
    # 网络参数初始化，注意pytorch有默认的初始化方法，也可以自定义初始化方法
    net.apply(init_nn)

    # 损失函数定义
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    epochs = 10
    for epoch in range(epochs):
        avg_loss = train_one_epoch(net, train_iter, trainer, loss)
        print(f'Epoch: {epoch + 1}/{epochs}, loss: {avg_loss}')
        acc = validate(net, test_iter)
        print(f'Test acc:{acc}')