import torch
import time
import torch.nn as nn
import torch.optim as optim
import copy
from torchvision import datasets
from torchvision import transforms
from RFC import calculate_all_layer_filter_index
from pruning_model import make_cfg_mask , make_newmodel
import numpy as np
from models import assistant_net
from models import resnet
import os
from thop import profile
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


parser = argparse.ArgumentParser(description='PRF........')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--num_workers', type=float, default=25, metavar='M',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weightdecay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--threshold', type=float, default=0.1,
                    help='threshold')
parser.add_argument('--balance', type=float, default=0.8,
                    help='balance')
parser.add_argument('--Restrictions', type=float, default=1953761888,
                    help='Restrictions (default: Restrictionstype=FLOPs)')
parser.add_argument('--netword_FLOPs', type=float, default=4111414272,
                    help='Restrictions (default: Restrictionstype=FLOPs)')
parser.add_argument('--Restrictionstype', type=str, default="FLOPs",
                    help='The type of restriction')

args = parser.parse_args()

def main(model = None,epochs=None ):
    best_prec1 = 0
    for epoch in range(0, epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 , woyongbudao, p3  = validate(val_loader, model, criterion)
        # remember best prec@1 and save checkpoint
        if prec1.cpu().numpy() > best_prec1:
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,"./output_model/resnet50_turntrain22.pth")
        best_prec1 = max(prec1.cpu().numpy(), best_prec1)
        print(best_prec1)
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    resnet50.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 1000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    list_target=[]
    # switch to evaluate mode
    model.eval()
    print("-------------------------------------------Use 50 batches for RFC")
    end = time.time()
    g1_list = []

    for i, (input, target) in enumerate(val_loader):
        if i in range(50):
            with torch.no_grad():
                target = target.cuda()
                input_var = input
                target_var = target
                list_target += target.tolist()
        # compute output
                output = model(input_var)
                g1 = nn.functional.softmax(output[0],dim=0).cpu().detach().tolist()
                g1_list.append(g1)
                loss = criterion(output, target_var)
        # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
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
        else:
            break
    print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        i, len(val_loader), batch_time=batch_time, loss=losses,
        top1=top1, top5=top5))
    return top1.avg , list_target , g1_list

def validate_g(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    list_target=[]
    g1_list = []
    # switch to evaluate mode
    model.eval()
    print("-------------------------------------------Use 50 batches for RFC")
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if i in range(50):
            with torch.no_grad():
                target = target.cuda()
                input_var = input
                target_var = target
                list_target += target.tolist()
        # compute output
                output = model(input_var)
                g1 = nn.functional.softmax(output[0],dim=0).cpu().detach().tolist()
                g1_list.append(g1)
                loss = criterion(output, target_var)
        # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
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
        else:
            break
    print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        i, len(val_loader), batch_time=batch_time, loss=losses,
        top1=top1, top5=top5))
    return top1.avg , list_target , g1_list

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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

criterion = nn.CrossEntropyLoss().cuda()

traindir = os.path.join("/datasets/ILSVRC2012/", 'train')
valdir = os.path.join("/datasets/ILSVRC2012/", 'val2')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers,pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.test_batch_size, shuffle=True,
    num_workers= args.num_workers ,pin_memory=True)

def delete_downsample(cfg):
    cfg.pop(4)
    cfg.pop(23)
    cfg.pop(42)
    return cfg
def add_downsample(cfg):
    cfg.insert(3, cfg[3])
    cfg.insert(23, cfg[22])
    cfg.insert(42, cfg[41])
    return cfg
def correct_index(index):
    correct = 0
    if index<=1:
        correct = index
    elif index<=3:
        correct = index+1
    elif index<=6:
        correct = index+2
    elif index<=9:
        correct = index+3
    elif index<=12:
        correct = index+4
    return correct
def compare_a_change(prec):
    sum=0
    for i in range(prec.size):
        if prec[i] > args.threshold:
            sum += 1
    return sum
def compare_a_change2(prec):
    sum=0
    for i in range(prec.size):
        if prec[i] > args.threshold:
            sum += 1
    return sum
def redundancy_set_shortcut(all_layer_mixent_index=None):
    prec = []
    for i in range(49):
        if i in [9,21,39,48]:
            if cfg[i]==1:
                prec.append(100)
                continue
            else:
                cff = cfg.copy()
                if i in [9,48]:
                    for o in range(3):
                        cff[i- o * 3] = int(cfg[i - o * 3] - (cfg[i - o * 3] * a / 3 + 1))
                elif i==21:
                    for o in range(4):
                        cff[i- o * 3] = int(cfg[i - o * 3] - (cfg[i - o * 3] * a / 4 + 1))
                elif i==39:
                    for o in range(6):
                        cff[i- o * 3] = int(cfg[i - o * 3] - (cfg[i - o * 3] * a / 6 + 1))
                print(cff)
                resnet50 = resnet.resnet50_official(cfg=cff).to(device)
                resnet50 = nn.DataParallel(resnet50, device_ids=[0, 1, 2, 3]).to(device)
                cfg_mask = make_cfg_mask(resnet56__p, all_layer_mixent_index, cfg=cff)
                make_newmodel(resnet56__p, resnet50, cfg_mask)
                prec1, woyongbudao, g1  = validate(train_loader, resnet50, criterion)
                prec.append((1-(cos_sim(g,g1)))*(1-args.balance)+(original_acc - prec1.cpu().numpy())*args.balance*0.1)
        else:
            prec.append(100)
    prec = np.squeeze(prec)
    return prec

def redundancy_set_nonshortcut(all_layer_mixent_index=None):
    prec = []
    for i in range(0,49):
        if i in [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]:
            prec.append(100)
        else:
            if cfg[i]==1:
                prec.append(100)
                continue
            else:
                cff = cfg.copy()
                cff[i] = int(cfg[i] - cfg[i] * b)
                print(cff)
                resnet50 = resnet.resnet50_official(cfg=cff).cuda()
                resnet50 = nn.DataParallel(resnet50, device_ids=[0, 1, 2, 3]).to(device)
                cfg_mask = make_cfg_mask(resnet56__p, all_layer_mixent_index, cfg=cff)
                make_newmodel(resnet56__p, resnet50, cfg_mask)
                prec1 ,woyongbudao, g1 = validate(train_loader, resnet50, criterion)
                prec.append((1-(cos_sim(g,g1)))*(1-args.balance)+(original_acc - prec1.cpu().numpy())*args.balance)
    prec = np.squeeze(prec)
    return prec

def cos_sim(vector_a, vector_b):
    con_sum=[]
    for i in range(len(vector_a)):
        for j in range(len(vector_a[i])):
            vector_a1 = np.mat(vector_a[i][j])
            vector_b1 = np.mat(vector_b[i][j])
            num = float(vector_a1 * vector_b1.T)
            denom = np.linalg.norm(vector_a1) * np.linalg.norm(vector_b1)
            sim = num / denom
            con_sum.append(sim)
    return sum(con_sum)/len(con_sum)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
kwargs = {'num_workers': 25, 'pin_memory': True} if use_gpu else {}

cfg=[64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]

resnet56__p = assistant_net.ResNet(cfg = cfg).to(device)
resnet56__p = nn.DataParallel(resnet56__p,device_ids=[0,1,2,3])

resnet56__p.load_state_dict(torch.load("./input/resnet50_p.pth"),strict=False)
test_loss = 0
correct = 0

#To get the attributes of the model
for i,j in enumerate(resnet56__p.modules()):
    if i == 1 :
        model = j

torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed(args.seed)

a = 0.5
b = 0.5
a_change = []
for index in range(10000):
    print("-------------------------------------------Prune the shortcut connection")
    # RFC
        # Get a collection of responses
    del model.P0,model.P1,model.P2,model.P3,model.P4,model.P5,model.P6,model.P7,model.P8,model.P9,\
        model.P10,model.P11,model.P12,model.P13,model.P14,model.P15,model.P16,model.P17,model.P18,model.P19,\
        model.P20,model.P21,model.P22,model.P23,model.P24,model.P25,model.P26,model.P27,model.P28,model.P29,\
        model.P30,model.P31,model.P32,model.P33,model.P34,model.P35,model.P36,model.P37,model.P38,model.P39,\
        model.P40,model.P41,model.P42,model.P43,model.P44,model.P45,model.P46,model.P47,model.P48,model.P1_1,model.P2_2,model.P3_3,model.P4_4

    for (i,j) in enumerate(resnet56__p.modules()):
        if i==1:
            for k in range(49):
                exec("j.P%s = []" % (k))
            j.P1_1 = []
            j.P2_2 = []
            j.P3_3 = []
            j.P4_4 = []

    OUT, list_target, g = validate_g(train_loader, resnet56__p, criterion)
    original_acc = OUT.cpu().numpy()
    print(original_acc)

    for i, j in enumerate(resnet56__p.modules()):
        if i == 1:
            model = j

    P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, \
    P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, \
    P47, P48, P49, P50, P51, P52 = model.P0, model.P1, model.P2, model.P3, model.P4, model.P5, model.P6, model.P7, model.P8, model.P9, \
    model.P10, model.P11, model.P12, model.P13, model.P14, model.P15, model.P16, model.P17, model.P18, model.P19, \
    model.P20, model.P21, model.P22, model.P23, model.P24, model.P25, model.P26, model.P27, model.P28, model.P29, \
    model.P30, model.P31, model.P32, model.P33, model.P34, model.P35, model.P36, model.P37, model.P38, model.P39, \
    model.P40, model.P41, model.P42, model.P43, model.P44, model.P45, model.P46, model.P47, model.P48, model.P1_1, model.P2_2, model.P3_3, model.P4_4

    for i in range(53):
        exec("P%s = np.concatenate(P%s)" % (i, i))

    P = [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23,
         P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45,
         P46,P47, P48, P49, P50, P51, P52]

    # The redundant set
    all_layer_mixent_index = calculate_all_layer_filter_index(P, list_target, cfg=cfg)
    prec = redundancy_set_shortcut(all_layer_mixent_index=all_layer_mixent_index)
    print(prec)

    # Determine whether to make a pruning granularity change
    if compare_a_change2(prec) == 49:
        a = 0.5 * a
        a_change.append(index)
        print("-------------------------------------------Change the granularity of pruning")
        print(a)
    else:
        # Look for the most redundant semantic layer
        for index,j in enumerate(prec):
            if j< args.threshold:
                if index in [9, 48]:
                    for o in range(3):
                        cfg[index - o * 3] = int(
                            cfg[index - o * 3] - (cfg[index - o * 3] * a / 3 + 1))
                elif index == 21:
                    for o in range(4):
                        cfg[index - o * 3] = int(
                            cfg[index - o * 3] - (cfg[index - o * 3] * a / 4 + 1))
                elif index == 39:
                    for o in range(6):
                        cfg[index - o * 3] = int(
                            cfg[index - o * 3] - (cfg[index - o * 3] * a / 6 + 1))
            print(cfg)

        resnet50 = resnet.resnet50_official(cfg=cfg).to(device)
        flops1, param1 = profile(resnet50, inputs=(torch.rand(1, 3, 224, 224).cuda(),))
        print(flops1)
        print(param1)

        resnet50 = nn.DataParallel(resnet50, device_ids=[0, 1, 2, 3]).to(device)
        cfg_mask = make_cfg_mask(resnet56__p, all_layer_mixent_index, cfg=cfg)
        make_newmodel(resnet56__p, resnet50, cfg_mask)

        optimizer = optim.SGD(resnet50.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weightdecay)
        main(model=resnet50, epochs=1)
        resnet50.load_state_dict(torch.load("./output_model/resnet50_turntrain22.pth"))
        resnet56__p = assistant_net.ResNet(cfg=cfg).to(device)
        resnet56__p = nn.DataParallel(resnet56__p, device_ids=[0, 1, 2, 3]).to(device)

        for [m0, m1] in zip(resnet50.modules(), resnet56__p.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
            elif isinstance(m0, nn.Conv2d):
                w1 = m0.weight.data[:, :, :].clone()
                w1 = w1[:, :, :].clone()
                m1.weight.data = w1.clone()
            elif isinstance(m0, nn.Linear):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
        #2941414272
        if args.Restrictionstype == "FLOPs":
            if flops1 < args.netword_FLOPs-(args.netword_FLOPs-args.Restrictions)*0.4:
                print(cfg)
                print(a_change)
                torch.save(resnet50.state_dict(), "./output_model/resnet50_shortcut.pth")
                break
        elif args.Restrictions_shortcut == "Pruningratio":
            if sum(cfg) < args.Restrictions_shortcut:
                print(cfg)
                print(a_change)
                torch.save(resnet50.state_dict(),"./output_model/resnet50_shortcut.pth")
                break


# DLRFC
for index in range(10000):
    print("-------------------------------------------Prune the non-shortcut connection")
    # RFC
        # Get a collection of responses
    del model.P0, model.P1, model.P2, model.P3, model.P4, model.P5, model.P6, model.P7, model.P8, model.P9, \
        model.P10, model.P11, model.P12, model.P13, model.P14, model.P15, model.P16, model.P17, model.P18, model.P19, \
        model.P20, model.P21, model.P22, model.P23, model.P24, model.P25, model.P26, model.P27, model.P28, model.P29, \
        model.P30, model.P31, model.P32, model.P33, model.P34, model.P35, model.P36, model.P37, model.P38, model.P39, \
        model.P40, model.P41, model.P42, model.P43, model.P44, model.P45, model.P46, model.P47, model.P48, model.P1_1, model.P2_2, model.P3_3, model.P4_4
    for (i,j) in enumerate(resnet56__p.modules()):
        if i==1:
            for k in range(49):
                exec("j.P%s = []" % (k))
            j.P1_1 = []
            j.P2_2 = []
            j.P3_3 = []
            j.P4_4 = []

    OUT, list_target, g = validate_g(train_loader, resnet56__p, criterion)
    original_acc = OUT.cpu().numpy()
    print(original_acc)
    for i, j in enumerate(resnet56__p.modules()):
        if i == 1:
            model = j
    P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, \
    P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, \
    P47, P48, P49, P50, P51, P52 = model.P0, model.P1, model.P2, model.P3, model.P4, model.P5, model.P6, model.P7, model.P8, model.P9, \
    model.P10, model.P11, model.P12, model.P13, model.P14, model.P15, model.P16, model.P17, model.P18, model.P19, \
    model.P20, model.P21, model.P22, model.P23, model.P24, model.P25, model.P26, model.P27, model.P28, model.P29, \
    model.P30, model.P31, model.P32, model.P33, model.P34, model.P35, model.P36, model.P37, model.P38, model.P39, \
    model.P40, model.P41, model.P42, model.P43, model.P44, model.P45, model.P46, model.P47, model.P48, model.P1_1, model.P2_2, model.P3_3, model.P4_4
    for i in range(53):
        exec("P%s = np.concatenate(P%s)" % (i, i))
    P = [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23,
         P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45,
         P46,P47, P48, P49, P50, P51, P52]

    # The redundant set
    all_layer_mixent_index = calculate_all_layer_filter_index(P, list_target, cfg=cfg)
    prec = redundancy_set_nonshortcut(all_layer_mixent_index=all_layer_mixent_index)
    print(prec)

    # Determine whether to make a pruning granularity change
    if compare_a_change(prec) == 49:
        b = 0.5 * b
        a_change.append(index)
        print("-------------------------------------------Change the granularity of pruning")
        print(b)
    else:
        for index,j in enumerate(prec):
            if j< args.threshold:
                print(cfg)
                cfg[index] = int(cfg[index] - cfg[index]*b)
                print(cfg)
        print(cfg)

        # Look for the most redundant semantic layer

        resnet50 = resnet.resnet50_official(cfg=cfg).to(device)

        flops1, param1 = profile(resnet50, inputs=(torch.randn(1, 3, 224, 224).cuda(),))
        print(flops1)
        print(param1)

        resnet50 = nn.DataParallel(resnet50, device_ids=[0, 1, 2, 3]).to(device)
        cfg_mask = make_cfg_mask(resnet56__p, all_layer_mixent_index, cfg=cfg)
        make_newmodel(resnet56__p, resnet50, cfg_mask)
        print(b)


        optimizer = optim.SGD(resnet50.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weightdecay)
        main(model=resnet50, epochs=1)
        resnet50.load_state_dict(torch.load("./output_model/resnet50_turntrain22.pth"))
        resnet56__p = assistant_net.ResNet(cfg=cfg).to(device)
        resnet56__p = nn.DataParallel(resnet56__p, device_ids=[0, 1, 2, 3]).to(device)
        for [m0, m1] in zip(resnet50.modules(), resnet56__p.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
            elif isinstance(m0, nn.Conv2d):
                w1 = m0.weight.data[:, :, :].clone()
                w1 = w1[:, :, :].clone()
                m1.weight.data = w1.clone()
            elif isinstance(m0, nn.Linear):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()

        if args.Restrictionstype == "FLOPs":
            if flops1 < args.Restrictions:
                print(cfg)
                print(a_change)
                torch.save(resnet50.state_dict(), "./output_model/resnet50_shortcut.pth")
                break
        elif args.Restrictions_shortcut == "Pruningratio":
            if sum(cfg) < args.Restrictions:
                print(cfg)
                print(a_change)
                torch.save(resnet50.state_dict(),"./output_model/resnet50_shortcut.pth")
                break


