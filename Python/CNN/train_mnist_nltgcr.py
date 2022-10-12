# -*- coding: utf-8 -*-


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import pickle



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc_list.append(100. * correct / len(test_loader.dataset))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device ='cuda'

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    loss_list = []
    test_acc_list = []



    def FF(x, y):
        # reload(w)
        y_pred = model(x)
        f = func(y_pred, y)
        f.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        vl = []
        for param in model.parameters():
            v = param.grad.view(-1)
            vl.append(v)
        fp = torch.cat(vl)  
        model.zero_grad()
        return fp
    
    def reload(fp):
        offset = 0
        for k, v in sizeG.items():
            G_dict[k].data.copy_(fp[offset: offset + v.numel()].view(v))
            offset = offset + v.numel()
        model.load_state_dict(G_dict) 

    def combine(net_G):
        G_dict = dict(net_G.state_dict())
        vl = []
        for key in G_dict:
            v = G_dict[key].view(-1)
            vl.append(v)
        fp = torch.cat(vl)    
        return fp
    
    func =  F.cross_entropy
    G_dict = dict(model.state_dict())
    sizeG = {}
    for key in G_dict:
        sizeG[key] = G_dict[key].shape
    w =  combine(model)
    sum_G = sum(v.numel() for _, v in G_dict.items())
    assert(len(w) ==sum_G)
    d = len(w)
    lb= 1
    # ep = 1e-15
    # real = torch.tensor([0], dtype = torch.float32)
    # imag = torch.tensor([1], dtype= torch.float32)
    
    # imagi = torch.complex(real, imag).to('cuda')
    x, y = next(iter(train_loader))
    P = torch.zeros((d, lb), requires_grad=False).to(device)
    AP = torch.zeros((d,lb), requires_grad=False).to(device)
    reload(w)
    r = FF(x.to(device), y.to(device))
    rho = torch.norm(r)
    epsf = 1
    ep = epsf * torch.norm(w)/rho
    ep = 1
    
    w1 = w-ep*r
    reload(w1)
    Ar = (FF(x.to(device), y.to(device))-r)/ep
    reload(w)
    t = torch.norm(Ar)
    t = 1.0/t
    
    P[:,0] = t * r
    AP[:,0]=  t * Ar 

    
    i2 = 1
    i = 1
    for epoch in range(1, args.epochs + 1):
           
        correct = 0
        model.train()
        for it, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if it % args.log_interval == 0:
                with torch.no_grad():
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, it * len(data), len(train_loader.dataset),
                        100. * it / len(train_loader), loss.item()))
                    loss_list.append(loss.item())
                    
            alph = AP.t()@r
            with torch.no_grad():
                dire = P@alph
                w = w + dire
                reload(w)
            r = FF(data, target)
            with torch.no_grad():
                rho = torch.norm(r)
                w1 = w-ep*r
                reload(w1)
            r1 = FF(data, target)
            Ar = (r1-r)/ep
            reload(w)
            if epoch < 5:
                ep = 1
            else:
                ep = epsf * rho/torch.norm(w1)
            p = r
            if i <= lb:
                k = 0
            else:
                k = i2
            while True:
                if k ==lb:
                    k = 0
                k +=1
                tau = torch.inner(Ar, AP[:,k-1])
                p = p - tau*(P[:,k-1])
                Ar = Ar -  tau*(AP[:,k-1])
                if k == i2:
                    break
            t = torch.norm(Ar)
            if (i2) == lb:
                i2 = 0
            i2 = i2+1
            i = i+1
            t = 1.0/t
            AP[:,i2-1] = t*Ar
            P[:,i2-1] = t*p
        test(model, device, test_loader)


    with open("results/loss_nltgcr5.pkl", "wb") as fp:  
        pickle.dump( loss_list, fp)
    with open("results/acc_nltgcr5.pkl", "wb") as fp:  
        pickle.dump( test_acc_list, fp)
