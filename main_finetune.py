import argparse
from ast import Break
import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import utils
from utils import common
from utils.common import adjust_learning_rate
from utils.evaluation import AverageMeter, accuracy
from models import resnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# print(torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

model_names = [ 'resnet50']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Fine-tuning')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=118, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', type=float, nargs='*', default=[0.06], metavar='LR',
                    help="the learning rate in each stage (default 1e-2, 1e-3)")
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[],
                    help="the epoch to decay the learning rate (default 0.5, 0.75)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument("--debug", action="store_true",
                    help="enable debug mode")
parser.add_argument('--seed', type=int, metavar='S', default=123,
                    help='random seed (default: 666)')
parser.add_argument('--lr-strategy', choices=['cos', 'step'],default='cos',
                    help='Learning rate decay strategy. \n'
                         '- cos: Cosine learning rate decay. In this case, '
                         '--lr should be only one value, and --decay-epoch will be ignored.\n'
                         '- step: Decay as --lr and --decay-step.')
parser.add_argument('--lighting', action='store_true',
                    help='[DEPRECATED] Use lighting in data augmentation.')
parser.add_argument('--warmup', action='store_true',
                    help='Use learning rate warmup in first five epochs. '
                         'Only available when --scratch is enabled.')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # if args.warmup and not args.scratch:
    #     raise ValueError("Finetuning should not use --warmup.")

    print(args)
    print(f"Current git hash: {common.get_git_id()}")

    args.distributed = args.world_size > 1

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    cfg=[64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]

    # from thop import profile

    args.arch == "resnet50"

    model = resnet.resnet50_official(cfg=cfg).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
    
    model.load_state_dict(torch.load("./output_model/resnet50_shortcut.pth"),strict=False)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr[0],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    cudnn.benchmark = True
    # Data loading code
    traindir = os.path.join("/datasets/ILSVRC2012/", 'train')
    valdir = os.path.join("/datasets/ILSVRC2012/", 'val2')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    training_transformers = [transforms.RandomResizedCrop(224)]
    if args.lighting:
        training_transformers.append(utils.common.Lighting(0.1))
    training_transformers += [transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize]

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(training_transformers))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, epoch=0, writer=None)
        return
    import copy
    print(model)
    prec1 = validate(val_loader, model, criterion, 1, writer=None)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # print(optimizer.state_dict()['param_groups'][0]['lr'])

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer=None, mask=None)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer=None)
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts,"./output_model/resnet50_turntrain2222.pth")
        best_prec1 = max(prec1, best_prec1)
        print(best_prec1)
    print("Best prec@1: {}".format(best_prec1))




def bn_weights(model):
    weights = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))

    return weights
    pass


def BN_grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)


def clamp_bn(model):
    bn_modules = list(filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))
    for m in bn_modules:
        m.weight.data.clamp_(0, 1)


def train(train_loader, model, criterion, optimizer, epoch, writer=None, mask=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # loss_aux_recorder = AverageMeter()
    # avg_sparsity_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch,
                             train_loader_len=len(train_loader), iteration=i,
                             decay_strategy=args.lr_strategy, warmup=args.warmup,
                             total_epoch=args.epochs, lr=args.lr, decay_epoch=args.decay_epoch)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        if isinstance(output, tuple):
            output, out_aux = output
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask finetuning style: do not actullay prune the network,
        # just simply disable the updating of the pruned layers
        if mask is not None:
            for name, p in model.named_parameters():
                if 'weight' in name:
                    p.grad.data = p.grad.data * mask[name]

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {3}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        if args.debug and i >= 5:
            break

    if writer:
        writer.add_scalar("train/cross_entropy", losses.avg, epoch)
        writer.add_scalar("train/top1", top1.avg.item(), epoch)
        writer.add_scalar("train/top5", top5.avg.item(), epoch)


def validate(val_loader, model, criterion, epoch, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            if isinstance(output, tuple):
                output, out_aux = output
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 200 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
            if args.debug and i >= 5:
                break

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if writer is not None:
        writer.add_scalar("val/cross_entropy", losses.avg, epoch)
        writer.add_scalar("val/top1", top1.avg.item(), epoch)
    return top1.avg

if __name__ == '__main__':
    main()
