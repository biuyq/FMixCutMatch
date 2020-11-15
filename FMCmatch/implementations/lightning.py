import torch.nn.functional as F
from fmix import sample_mask, FMixBase
import torch
import numpy as np

def fmix_loss(input, y1, y_cut1, y_cut2, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix

    Args:
        input: If train, mixed input. If not train, standard input
        y1: Targets for first image
        index: Permutation for mixing
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """

    if train and not reformulate:
       
        #onehot = torch.zeros(100,10).cuda()
        #y1 = onehot.scatter_(1,y1.unsqueeze(1),1) 

        y2 = y1[index]
        if lam >0.5: 
            ycut = y1 
            y_cut = y_cut1
            lama = lam

        else: 
            ycut = y2
            y_cut = y_cut2
            lama = 1-lam

        #neighbor = torch.eq(torch.argmax(y1, dim=1),torch.argmax(y2, dim=1)).type(torch.cuda.FloatTensor)

  

        y1_l = -torch.mean(torch.sum(y1 * F.log_softmax(input, dim=1), dim=1) )
        y2_l = -torch.mean(torch.sum(y2 * F.log_softmax(input, dim=1), dim=1) )


        #y1_l = torch.mean(torch.sum((y1 - torch.softmax(input, dim=1))**2, dim=1) )
        #y2_l = torch.mean(torch.sum((y2 - torch.softmax(input, dim=1))**2, dim=1) )
        #cut_loss = -torch.mean(torch.sum(ycut * F.log_softmax(y_cut, dim=1), dim=1)) * lama
        cut1_loss = -torch.mean(torch.sum(y1 * F.log_softmax(y_cut1, dim=1), dim=1)) * lam
        cut2_loss = -torch.mean(torch.sum(y2 * F.log_softmax(y_cut2, dim=1), dim=1)) * (1-lam)

        #print(torch.softmax(y_cut, dim = 1))
        return     y1_l * lam + y2_l * (1 - lam) + cut1_loss + cut2_loss
        #return F.cross_entropy(input, y1) * lam + F.cross_entropy(input, y2) * (1 - lam)
    else:
        return -torch.mean(torch.sum(y1 * F.log_softmax(input, dim=1), dim=1))


def fmix_loss2(input, y1, y_cut1, y_cut2, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix

    Args:
        input: If train, mixed input. If not train, standard input
        y1: Targets for first image
        index: Permutation for mixing
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """

    if train and not reformulate:

        y2 = y1[index]
        #if lam >0.5: 
        #    ycut = y1 
        #    lama = lam
        #else: 
        #    ycut = y2
        #    lama = 1-lam

        y1_l = -torch.mean(torch.sum(y1 * F.log_softmax(input, dim=1), dim=1))
        y2_l = -torch.mean(torch.sum(y2 * F.log_softmax(input, dim=1), dim=1))

        cut1_loss = -torch.mean(torch.sum(y1 * F.log_softmax(y_cut1, dim=1), dim=1)) * lam
        cut2_loss = -torch.mean(torch.sum(y2 * F.log_softmax(y_cut2, dim=1), dim=1)) * (1-lam)
        #print(cut_loss*lama)
        return y1_l * lam + y2_l * (1 - lam) + ( cut1_loss + cut2_loss)  #
        #return F.cross_entropy(input, y1) * lam + F.cross_entropy(input, y2) * (1 - lam)
    else:
        return -torch.mean(torch.sum(y1 * F.log_softmax(input, dim=1), dim=1))

class FMix(FMixBase):
    r""" FMix augmentation

        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].

        Example
        -------

        .. code-block:: python

            class FMixExp(pl.LightningModule):
                def __init__(*args, **kwargs):
                    self.fmix = Fmix(...)
                    # ...

                def training_step(self, batch, batch_idx):
                    x, y = batch
                    x = self.fmix(x)

                    feature_maps = self.forward(x)
                    logits = self.classifier(feature_maps)
                    loss = self.fmix.loss(logits, y)

                    # ...
                    return loss
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def __call__(self, x):
        # Sample mask and generate random permutation
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        # Mix the images
        x1 = mask * x
        x2 = (1 - mask) * x[index]
        self.index = index
        self.lam = lam

        if lam >= 0.5:
            x_cut = x1
        else:
            x_cut = x2

        
        masky = mask.cpu()
        #print(masky.numpy())
        #np.save("mask.txt",masky.numpy())
        return x1+x2, x1, x2


    def loss(self, y_pred, y_cut1, y_cut2, y, epoch, train=True):
        
        return fmix_loss(y_pred,  y, y_cut1, y_cut2, self.index, self.lam, train, self.reformulate)

