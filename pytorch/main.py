#!/usr/bin/env python

# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import mnist
import os


n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100

data = mnist.MNIST("../data/")
data.gz = True

train_image, train_label = data.load_training()
test_image, test_label = data.load_testing()
train_size = len(train_image)
test_size = len(test_image)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = np.array(self.images[idx])
        img = np.reshape(img, (28, 28))
        # print(img.shape)
        img = np.expand_dims(img, axis=-1)
        # print(img.shape)
        if self.trans:
            img = self.trans(img)
        return img, self.labels[idx]


train_loader = torch.utils.data.DataLoader(Dataset(train_image, train_label,
                                                   trans=torchvision.transforms.ToTensor()),
                                           batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(Dataset(test_image, test_label,
                                                  trans=torchvision.transforms.ToTensor()),
                                          batch_size=batch_size_test, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, training=True):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.dropout = torch.nn.Dropout2d(p=0.2)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.dropout(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


network = Net()
optim = torch.optim.SGD(network.parameters(),
                        lr=learning_rate,
                        momentum=momentum)
loss_object = torch.nn.NLLLoss()

train_losses = []
test_losses = []


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optim.zero_grad()
        output = network(data.float())
        loss = loss_object(output, target)
        loss.backward()
        optim.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
    torch.save(network.state_dict(), "./results/model_e{}.pth".format(epoch))


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.float())
            test_loss += loss_object(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))


if not os.path.exists("./results"):
    os.mkdir("./results")
test()
for epoch in range(1, n_epochs+1):
    train(epoch)
    test()

np.save('./results/train_loss.npy', train_losses)
np.save('./results/test_loss.npy', test_losses)
