import torchvision
from torchvision import transforms
from torch.utils import data
from torchvision import datasets
import torch
from torch import nn

def data_downloader(batch_size):
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

    return train_iter, test_iter

def init_nn(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_model(net, train_loader, optimizer, lossfun):
    net.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = net(imgs)
        loss = lossfun(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(train_loader)
    return avg_loss

def eval_model(net, test_loader):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
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

    BATCH_SIZE = 256
    EPOCHS = 10
    train, test = data_downloader(BATCH_SIZE)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(28*28, 256), nn.ReLU(),
                        nn.Linear(256, 10)
                        ).to(device)
    net.apply(init_nn)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(EPOCHS):
        train_loss = train_model(net, train, optimizer, loss_fun)
        print(f'Epoch: {epoch + 1}/{EPOCHS}, loss: {train_loss},', end=' ')
        acc = eval_model(net, test)
        print(f'Test acc:{acc}')
