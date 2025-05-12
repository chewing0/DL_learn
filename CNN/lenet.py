import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision import datasets

# 数据集加载
def data_download_fashionmnist(batch_size):
    # 下载数据集
    train_dataset = datasets.FashionMNIST(root='learning/data',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5), (0.5)),
                                          ])
                                          )
    test_dataset = datasets.FashionMNIST(root='learning/data',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5), (0.5)),
                                         ])
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

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    LR = 0.1

    train, test = data_download_fashionmnist(BATCH_SIZE)

    net = LeNet().to(device)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(net, train, loss_fun, optimizer, device)
        print(f'Epoch: {epoch + 1}/{EPOCHS}, loss: {train_loss},', end=' ')
        acc = eval_model(net, test, device)
        print(f'Test acc:{acc}')
