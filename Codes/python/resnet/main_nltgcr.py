# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    model_names = sorted(name for name in resnet.__dict__
        if name.islower() and not name.startswith("__")
                         and name.startswith("resnet")
                         and callable(resnet.__dict__[name]))
    
    print(model_names)
    
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet32)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    best_prec1 = 0
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.manual_seed(0)
    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.half:
        model.half()
        criterion.half()
    
    
    if args.evaluate:
        validate(val_loader, model, criterion)
    

    def FF(x, y):
        # reload(w)
        y_pred = model(x)
        f = criterion(y_pred, y)
        f.backward()
        vl = []
        for param in model.parameters():
            if param.grad is not None:
                vv = nn.functional.normalize(param.grad, p=2.0, dim = 0)
                v = vv.view(-1)
                vl.append(v)
                fp = torch.cat(vl)  
        model.zero_grad()
        return fp
    
    def reload(fp):
        offset = 0
        for name, param in model.named_parameters():
            shape = param.shape
            param.data = fp[offset: offset + shape.numel()].view(shape)
            offset = offset + shape.numel()
            
    def combine(model):
        vl = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                v = param.data.view(-1)
                vl.append(v)
        fp = torch.cat(vl)    
        return fp
    
    G_dict = {}
    for name in model.named_parameters():
        G_dict[name[0]] = name[1].data
  
    sizeG = {}
    for key in G_dict:
        sizeG[key] = G_dict[key].shape
    w =  combine(model)
    sum_G = sum(v.numel() for _, v in G_dict.items())
    assert(len(w) ==sum_G)
    d = len(w)
    lb= 1
    device = 'cuda'
    P = torch.zeros((d, lb), requires_grad=False).to(device)
    AP = torch.zeros((d,lb), requires_grad=False).to(device)
    reload(w)
    trainX, trainY = next(iter(train_loader))
    r = FF(trainX.to(device), trainY.to(device))
    rho = torch.norm(r)
    epsf = 1e-1
    ep = epsf * rho/torch.norm(w)
    w1 = w-ep*r
    reload(w1)
    Ar = (FF(trainX.to(device), trainY.to(device))-r)/ep
    reload(w)
    t = torch.norm(Ar)
    t = 1.0/t
    P[:,0] = t * r
    AP[:,0]=  t * Ar 
    loss_list = []
    i2 = 1
    i = 1
    lossL = []
    accL = []
    
    for epoch in range(args.start_epoch, args.epochs):
        """
            Run one train epoch
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
    
        # switch to train mode
        model.train()
    
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
    
            # measure data loading time
            data_time.update(time.time() - end)
    
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
            if args.half:
                input_var = input_var.half()
            # NLTGCR
            alph = AP.t()@r
            with torch.no_grad():
                dire = P@alph
                w = w + dire
                reload(w)
            r = FF(input_var, target_var)
            with torch.no_grad():
                rho = torch.norm(r)
                w1 = w-ep*r
                reload(w1)
            r1 = FF(input_var, target_var)
            Ar = (r1-r)/ep
            reload(w)
            
            ep = epsf * rho/torch.norm(w)
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
            with torch.no_grad():
                output = model(input_var)
                loss = criterion(output, target_var)
                output = output.float()
                loss = loss.float()
                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
        
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        
                if i % args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                              epoch, i, len(train_loader), batch_time=batch_time,
                              data_time=data_time, loss=losses, top1=top1))
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        lossL.append(losses.avg)
        accL.append(prec1)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
    
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))
    
    file = "results/loss_T"+str(6)+".pkl"
    with open(file, "wb") as fp:  
        pickle.dump(lossL, fp)
    file = "results/acc_T"+str(6)+".pkl"
    with open(file, "wb") as fp:  
        pickle.dump(accL, fp)