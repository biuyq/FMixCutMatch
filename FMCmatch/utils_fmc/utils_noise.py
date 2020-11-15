from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt
from utils.AverageMeter import AverageMeter
from utils.criterion import *
import sys
sys.path.append('../implementations/')
from lightning import FMix
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing as preprocessing


from tqdm import tqdm

from math import pi
from math import cos

##############################################################################
############################# TRAINING LOSSSES ###############################
##############################################################################

def loss_soft_reg_ep(preds, labels, soft_labels, device, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    #onehot = torch.zeros(args.batch_size,args.num_classes).cuda()
    #soft_labels = onehot.scatter_(1,soft_labels.unsqueeze(1),1) 

    L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))   # Soft labels
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    loss = L_c + args.alpha * L_p + args.beta * L_e
    return prob, loss

##############################################################################

def cyclic_lr(args, iteration, current_epoch, it_per_epoch):
    # proposed learning late function
    T_iteration = it_per_epoch*current_epoch + iteration
    T_epoch_per_cycle = args.SE_epoch_per_cycle*it_per_epoch
    T_iteration = T_iteration%T_epoch_per_cycle
    return args.lr * (cos(pi * T_iteration / T_epoch_per_cycle) + 1) / 2
##############################################################################

##############################################################################
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    #print(torch.max(y_b,dim=1)[0])
    #lama = torch.exp((torch.max(y_b,dim=1)[0]))/(torch.exp(torch.max(y_a,dim=1)[0])+torch.exp(torch.max(y_b,dim=1)[0]))
    #print(lama.size())
    #print(lama)
    #x_ = x[index, :]
    #print(args.batch_size)
    #mixed_x = x
    #for i in range(0,lama.size()[0]):
        #print(i)
        #mixed_x[i] = lama[i] * x[i,:] + (1 - lama[i]) * x_[i,:]
    return mixed_x, y_a, y_b, lam

def loss_mixup_reg_ep(preds_org_la, preds, labels, targets_a, targets_b, device, lam, args):   

    #a = math.floor(args.batch_size/args.labeled_batch_size/2)
    #preds_a= preds_org[0:a*args.labeled_batch_size,:]

    #preds_b = preds_org[-args.labeled_batch_size:,:].repeat(a,1)
    #print(preds_b.size())
   
    #preds_org_l = preds_org_la[0:a*args.labeled_batch_size,:]
    #preds_org2_l = labels[-args.labeled_batch_size:].repeat(1,a)
    #preds_a, preds_b = preds_org, preds_org2
    #print(math.ceil(args.batch_size/2))
    #print(preds_a.size())
    #print(preds_b.size())
    #print(targets_a)
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    #p_pre = torch.ones(args.num_classes).to(device) / args.num_classes
    #print(p_pre)
    #p = torch.tensor(prob_avg_all,dtype = torch.float32)[0].to(device) 
    #thrd = (1./args.num_classes)+( (1-(1./args.num_classes))* (epoch/args.epoch))
    #max_a = torch.max(targets_a, dim=1)[0]
    #mask_a_ = (max_a==1).type(torch.cuda.FloatTensor)
    #mask_a__ = (0.9>=max_a).type(torch.cuda.FloatTensor)
    #mask_a = (max_a>=thrd).type(torch.cuda.FloatTensor)
    #mask_a = mask_a*mask_a__+mask_a_

    #print(torch.sum(mask_a))
    #mask_a = torch.where(((max_a>=thrd))|(max_a==1.0), max_a, torch.zeros_like(max_a)).type(torch.cuda.FloatTensor)
    #max_b = torch.max(targets_b, dim=1)[0]
    #mask_b_ = (max_b==1).type(torch.cuda.FloatTensor)
    #mask_b__ = (0.9>=max_b).type(torch.cuda.FloatTensor)
    #mask_b = (max_a>=thrd).type(torch.cuda.FloatTensor)
    #mask_b = mask_b*mask_b__+mask_b_
    #mask_b = torch.where(((max_b>=thrd))|(max_b==1.0), max_b, torch.zeros_like(max_b)).type(torch.cuda.FloatTensor)

    #mask = max_b*mask_b
    #print(torch.sum(mask_a))
    #print(torch.sum(mask_b))
    #print(torch.sum(prob))
    #p = torch.tensor([4948,13861,10585,8497,7458,6882,5727,5595,5045,4659],dtype = torch.float).to(device) / 73257.0
    #n1 = torch.mean(preds_a, dim=1)
    #print(n1.size())
    #n2 = torch.mean(preds_b, dim=1)

    #thrd = 0.9 * torch.ones(args.batch_size).cuda()

    #max_a = torch.max(F.softmax(preds_org_l), dim=1)[0]
    #max_b = torch.max(F.softmax(preds_org2_l), dim=1)[0]
    #max_b = preds_org2_l
    #print(max_b)
    #print((max_a>=0.9).size())
    #neighbor = torch.eq(torch.argmax(targets_a, dim=1),torch.argmax(targets_b, dim=1)).type(torch.cuda.FloatTensor)
    #thdd = ((max_a>=0.1)).type(torch.cuda.FloatTensor)
    #print(thdd)
    #print(neighbor)

    #Ze = torch.zeros(a * args.labeled_batch_size).cuda()    
    #Ze = torch.zeros(args.batch_size).cuda()    
    #Ze = torch.zeros(math.ceil(args.batch_size/2)).cuda()
    #print(torch.max((n1-n2)**2))
    #pos = neighbor * torch.max(Ze,(n1-n2)**2) * thdd
    #neg = (1.-neighbor) * torch.max(Ze,0.1-(n1-n2)**2) * thdd
 
    #print(torch.max(Ze,(n1-n2)**2))
    #print(neighbor * torch.max(Ze,(n1-n2)**2)*thdd)
    #VAFM = (torch.sum(pos)/(torch.sum(neighbor*thdd)+1e-10)+torch.sum(neg)/(torch.sum((1.-neighbor) * thdd)+1e-10))/2.
    #VAFM = torch.mean(pos+neg)
    #onehot = torch.zeros(args.batch_size,args.num_classes).cuda()
    #targets_a = onehot.scatter_(1,targets_a.unsqueeze(1),1)   

    #onehot = torch.zeros(args.batch_size,args.num_classes).cuda()
    #targets_b = onehot.scatter_(1,targets_b.unsqueeze(1),1) 
    #targets_a_ = ((1/(torch.sum((p_pre/p)*targets_a, dim = 1)+1e-10))*(((p_pre/p)*targets_a).t())).t()
    #targets_b_ = ((1/(torch.sum((p_pre/p)*targets_b, dim = 1)+1e-10))*(((p_pre/p)*targets_b).t())).t()
    #print(torch.sum(targets_a_))
    #targets_a_ = ((((p_pre/p)*targets_a).t())/torch.sum((p_pre/p)*targets_a, dim = 1)).t()
    #targets_b_ = ((((p_pre/p)*targets_b).t())/torch.sum((p_pre/p)*targets_b, dim = 1)).t()
    #print(torch.sum((p_pre/p)*targets_a, dim = 1))    
    #print(torch.sum(targets_a))
    #print((p_pre/p).t().t())
 
    #mixup_loss_a = -torch.sum(mask_a*torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1))/torch.sum(mask_a) # needmask
    #mixup_loss_b = -torch.sum(mask_b*torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1))/torch.sum(mask_b)

    mixup_loss_a = -torch.mean(torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1)) # need no mask
    mixup_loss_b = -torch.mean(torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1))

    #mixup_loss_a = torch.mean(torch.sum((targets_a - torch.softmax(preds, dim=1))**2, dim=1) )
    #mixup_loss_b = torch.mean(torch.sum((targets_b - torch.softmax(preds, dim=1))**2, dim=1) )

    mixup_loss = lam * mixup_loss_a + (1-lam) * mixup_loss_b 
    #mixup_loss = torch.mean(lam * mixup_loss_a) + torch.mean((1 - lam) * mixup_loss_b)
    #onehot = torch.zeros(args.batch_size,args.num_classes).cuda()
    #print(mixup_loss)
    #print(labels)
    #one_hot = onehot.scatter_(1,labels.unsqueeze(1),1)
    #one_hot = zeros.scatter_(1, labels, 1)
    #CE_loss = -torch.mean(torch.sum(one_hot * F.log_softmax(preds_org_la, dim=1), dim=1))
    if args.lr >=0.1:
        alpha = args.alpha
    else:
    	alpha = args.alpha * args.lr

    if (args.dataset == 'svhn') & (args.labeled_batch_size <=16):
        alpha = 0.0
    #alpha = args.alpha
    L_p = -1./torch.mean(torch.log(prob_avg) * prob_avg)
    #print(torch.sum(p))
    #print(((p+1)/(args.num_classes+1)))
    #L_p = -torch.sum(torch.log(prob_avg) * p_pre)
    #L_p = -torch.sum(torch.log(p_pre/prob_avg) * p_pre)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))
    #print(L_e) 
    loss =  mixup_loss   + args.beta * L_e + alpha * L_p#+ args.gamma * VAFM  # + CE_loss  
    return prob, loss


##############################################################################

def train_CrossEntropy_partialRelab(args, model, device, train_loader, optimizer, epoch, train_noisy_indexes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()

####################################FMix##################################
    fmix = FMix(size=(args.img_size, args.img_size))


    top1 = AverageMeter()
    top5 = AverageMeter()
    w = torch.Tensor([0.0])

    top1_origLab = AverageMeter()

    # switch to train mode
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []

    alpha_hist = []

    end = time.time()

    results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)


    if args.loss_term == "Reg_ep":
        print("Training with cross entropy and regularization for soft labels and for predicting different classes (Reg_ep)")
    elif args.loss_term == "MixUp_ep":
        print("Training with Mixup and regularization for soft labels and for predicting different classes (MixUp_ep)")
        alpha = args.Mixup_Alpha

        print("Mixup alpha value:{}".format(alpha))

    target_original = torch.from_numpy(train_loader.dataset.original_labels)

    counter = 1
    for imgs, img_pslab, labels, soft_labels, index in train_loader:
        images = imgs.to(device)
        labels = labels.to(device)
        
        soft_labels = soft_labels.to(device)
        #print(torch.sum(soft_labels))
        if args.DApseudolab == "False":
            if device=='cuda':
                indexi = torch.randperm(args.batch_size).cuda()
            else:
                indexi = torch.randperm(args.batch_size)
            images_pslab = img_pslab.to(device)
            #images_pslab2 = images_pslab[indexi]
        if args.loss_term == "MixUp_ep":
            if args.dropout > 0.0 and args.drop_extra_forward == "True":
                if args.network == "PreactResNet18_WNdrop":
                    tempdrop = model.drop
                    model.drop = 0.0

                elif args.network == "WRN28_2_wn":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            tempdrop = m.p
                            m.p = 0.0
                else:
                    tempdrop = model.drop.p
                    model.drop.p = 0.0

            if args.DApseudolab == "False":
                optimizer.zero_grad()
                output_x1= model(images_pslab)

                output_x1.detach_()

                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                output_x1 = model(images)
                output_x1.detach_()

                optimizer.zero_grad()

            if args.dropout > 0.0 and args.drop_extra_forward == "True":
                if args.network == "PreactResNet18_WNdrop":
                    model.drop = tempdrop

                elif args.network == "WRN28_2_wn":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            m.p = tempdrop
                else:
                    model.drop.p = tempdrop
            images_org = images
            x_FMix, x_cut1, x_cut2 = fmix(images)
            images, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device)

        # compute output
        outputs = model(images)
        FM = model(x_FMix)
      
        outputs_org = model(images_org)
        ycut1 = model(x_cut1)
        ycut2 = model(x_cut2)

        loss_FM = fmix.loss(FM, ycut1, ycut2, soft_labels, epoch)
        if args.loss_term == "Reg_ep":
            prob, loss = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)

        elif args.loss_term == "MixUp_ep":
            prob = F.softmax(output_x1, dim=1)
            prob_mixup, loss = loss_mixup_reg_ep(outputs_org, outputs, labels, targets_a, targets_b, device, lam, args)
            outputs = output_x1
        #print(loss_FM)
        if args.use_FMix == 'True':
            loss += loss_FM
        #print(loss)
        results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 1])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        top1_origLab_avg = 0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if counter % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                       prec1, optimizer.param_groups[0]['lr']))
        counter = counter + 1

    if args.swa == 'True':
        if epoch > args.swa_start and epoch%args.swa_freq == 0 :
            optimizer.update_swa()
    #optimizer.swap_swa_sgd()
    # update soft labels

    train_loader.dataset.update_labels_randRelab(results, train_noisy_indexes, args.label_noise)

    return train_loss.avg, top5.avg, top1_origLab_avg, top1.avg, batch_time.sum

###################################################################################


def testing(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    with open("abliation_cifar100_10000_mini_off","a") as f:
        print ('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)), file = f)
    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]
    
    return (loss_per_epoch, acc_val_per_epoch)


def validating(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, _, target, _, _, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)
