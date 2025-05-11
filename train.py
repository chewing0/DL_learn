import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision import datasets
from net import LeNet, AlexNet, MLP, VGG11, NiN, GoogLeNet

# 数据集加载
def data_download_fashionmnist(batch_size, resize=None):
    t = [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))]
    if resize:
        t.insert(0, transforms.Resize(resize))
    # 下载数据集
    train_dataset = datasets.FashionMNIST(
                                          root=r'learning\data',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose(t)
                                          )
    test_dataset = datasets.FashionMNIST(
                                         root=r'learning\data',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose(t)
                                         )
    # 创建数据加载器
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 )
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 )
    return train_iter, test_iter

def train_one_epoch(net, train_data, loss_fun, optimizer, device):
    net.train()
    total_loss = 0
    for imgs, labels in train_data:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
    avg_loss = total_loss/len(train_data)
    return avg_loss

def eval_model(net, test_data, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_data:
            imgs, labels = imgs.to(device), labels.to(device)
            output = net(imgs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    acc = 100 * correct / total
    return acc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.01
    
    # 网络定义
    # net = MLP().to(device)
    # RESIZE = None
    # net = LeNet().to(device)
    # RESIZE = None
    # net = AlexNet().to(device)
    # RESIZE = 224
    # net = VGG11().to(device)
    # RESIZE = 224
    # net = NiN().to(device)
    # RESIZE = 224
    net = GoogLeNet().to(device)
    RESIZE = 224

    train, test = data_download_fashionmnist(BATCH_SIZE, resize=RESIZE)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(net, train, loss_fun, optimizer, device)
        print(f'Epoch: {epoch + 1}/{EPOCHS}, loss: {train_loss},', end=' ')
        acc = eval_model(net, test, device)
        print(f'Test acc:{acc}')