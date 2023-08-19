'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle
from resnet import ResNet18
from vgg import VGG
from utils import progress_bar
from method import NLTGCR



device = 'cuda' 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


print('==> Building model..')
net = ResNet18()
net = net.to(device)
cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
# optimizer = NLTGCR(
#                  net.parameters(),
#                  lr= 0.5,
#                  max_iter=1,
#                  max_eval=999,
#                  restart=100,
#                  history_size=5,
#                  strategy=None,
#             )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        def closure(nargout=1):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if nargout == 1:
                return loss
            else:
                return loss, outputs
        optimizer.step(closure)
        # with torch.no_grad():
        loss, outputs = closure(nargout=2)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), 100.*correct/total


loss_list = []
val_acc_list = []
rr = 1
for epoch in range(start_epoch, start_epoch+100):
    los, va = train(epoch)
    scheduler.step()
    loss_list.append(los)
    val_acc_list.append(va)

with open(f"cresults/loss_mom{rr}.pkl", "wb") as fp:  
    pickle.dump(loss_list, fp)

with open(f"cresults/acc_mom{rr}.pkl", "wb") as fp:  
    pickle.dump(val_acc_list, fp)