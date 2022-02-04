import sys
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import TensorDataset, SubsetRandomSampler
import numpy as np
from torchvision.datasets import FashionMNIST
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data.dataset import Dataset



class ANet(nn.Module):
    def __init__(self, image_size):
        super(ANet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class CNet(nn.Module):
    def __init__(self, image_size):
        super(CNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, 0.5, True, True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, True, True)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class DNet(nn.Module):
    def __init__(self, image_size):
        super(DNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn0 = nn.BatchNorm1d(100)
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class ENet(nn.Module):
    def __init__(self, image_size):
        super(ENet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.log_softmax(x)


class FNet(nn.Module):
    def __init__(self, image_size):
        super(FNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return F.log_softmax(self.fc5(x), dim=1)


def train(epoch, model, train_loader, train_str):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels.type(torch.int64))
        train_loss += F.nll_loss(output, labels, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.sampler)
    # print("train loss:" , train_loss)
    train_str.append(train_loss)
    # train_str.append(100. * correct / len(train_loader.sampler))


def test(model, test_loader, val_str, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            val_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    val_loss /= len(test_loader.sampler)
    # print("val loss:" , val_loss)
    val_str.append(val_loss)
    # val_str.append(100. * correct / len(test_loader.sampler))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})%)\n'.format(
        test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))

class createDataSet(Dataset):
    def __init__(self, x, transform=None):
        self.x = x / 255
        self.x = self.x.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        temp = self.x[idx]
        temp = temp.reshape(1, 784)
        if self.transform:
            temp = self.transform(temp)
        return temp


def realTest(model, test_loader):
    list = []
    model.eval()
    i = 0
    with torch.no_grad():
        for data in test_loader:
            i += 1
            output = model(data)
            prediction = output.max(1, keepdim=True)[1]
            for pred in prediction.data:
                list.append(pred.item())

        with open("test_y", "w+") as pred:
            pred.write('\n'.join(str(v) for v in list))


if __name__ == '__main__':
    my_x = sys.argv[1]
    my_y = sys.argv[2]
    my_test = np.loadtxt(sys.argv[3])

    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1,), (0.3,))])

    test_x = createDataSet(my_test, transform=transforms)
    real_test_loader = torch.utils.data.DataLoader(dataset=test_x, batch_size=64, shuffle=False)

    BATCH_SIZE = 64
    lr = 0.1
    train_set = FashionMNIST(root='./data',
                             train=True,
                             transform=transforms,
                             download=True)

    indices = list(range(len(train_set)))
    split = int(len(train_set) * 0.2)

    val_idx = np.random.choice(indices,
                               size=split,
                               replace=False)

    train_idx = list(set(indices) - set(val_idx))

    train_sam = SubsetRandomSampler(train_idx)
    val_sam = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=64,
                                               sampler=train_sam)
    val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                             batch_size=64,
                                             sampler=val_sam)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False,
                              download=True,
                              transform=transforms), batch_size=64, shuffle=True)

    model = DNet(image_size=28 * 28)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_str = []
    val_str = []
    for epoch in range(0, 10):
        train(epoch, model, train_loader, train_str)
        test(model, val_loader, val_str, epoch)
    realTest(model,real_test_loader)