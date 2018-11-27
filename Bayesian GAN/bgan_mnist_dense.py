from __future__ import print_function
import os, pickle
import numpy as np
import random, math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from statsutil import AverageMeter, accuracy
from matplotlib.ticker import MultipleLocator ,FormatStrFormatter
import matplotlib.pyplot as plt

# Default Parameters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=784)
parser.add_argument('--batchSize', type=int, default= 60, help='input batch size')
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default= 2 * 1e-4, help='learning rate, default=0.0002')
parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
parser.add_argument('--dim_h', type=int, default=128, help='number of GPUs to use')
parser.add_argument('--outf', default='modelfiles/MNIST_dense', help='folder to output images and model checkpoints')
parser.add_argument('--numz', type=int, default=1, help='The number of set of z to marginalize over.')
parser.add_argument('--num_mcmc', type=int, default=10, help='The number of MCMC chains to run in parallel')
parser.add_argument('--num_semi', type=int, default=4000, help='The number of semi-supervised samples')
parser.add_argument('--gnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--dnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--d_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--g_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--stats_interval', type=int, default= 600, help='Calculate test accuracy every interval')
parser.add_argument('--tensorboard', type=int, default=1, help='')
parser.add_argument('--bayes', type=int, default=1, help='Do Bayesian GAN or normal GAN')
import sys;

sys.argv = [''];
del sys
opt = parser.parse_args()
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)


# First, we construct the data loader for full training set
# as well as the data loader of a partial training set for semi-supervised learning
# transformation operator
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_opt = transforms.Compose([
    transforms.ToTensor(),
    #normalize,
])
# get training set and test set
dataset = MNIST(root='./data', download=True,
                     train=True,
                     transform=transform_opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True)

from partial_dataset import PartialDataset

# partial dataset for semi-supervised training
dataset_partial = PartialDataset(dataset, opt.num_semi)

# test set for evaluation
dataset_test = MNIST(root='./data',
                          train=False,
                          transform=transform_opt)
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=opt.batchSize, shuffle=False, pin_memory=True)

dataloader_semi = torch.utils.data.DataLoader(dataset_partial, batch_size=opt.batchSize,
                                              shuffle=True)

# Now we initialize the distributions of G and D
##### Generator ######
# opt.num_mcmc is the number of MCMC chains that we run in parallel
# opt.numz is the number of noise batches that we use. We also use different parameter samples for different batches
# we construct opt.numz * opt.num_mcmc initial generator parameters
# We will keep sampling parameters from the posterior starting from this set
# Keeping track of many MCMC chains can be done quite elegantly in Pytorch
from models.discriminators import dense_D
from models.generators import dense_G
from statsutil import weights_init

netGs = []
for _idxz in range(opt.numz):
    for _idxm in range(opt.num_mcmc):
        netG = dense_G(opt.dim_h, nz=opt.nz)
        # netG.apply(weights_init)
        netGs.append(netG)
##### Discriminator ######
# We will use 1 chain of MCMCs for the discriminator
# The number of classes for semi-supervised case is 11; that is,
# index 0 for fake data and 0-9 for the 10 classes of MNIST.
num_classes = 11
netD = dense_D(opt.dim_h, num_classes=num_classes)

# In order to calculate errG or errD_real, we need to sum the probabilities over all the classes (1 to K)
# ComplementCrossEntropyLoss is a loss function that performs this task
# We can specify a default except_index that corresponds to a fake label. In this case, we use index=0
from ComplementCrossEntropyLoss import ComplementCrossEntropyLoss

criterion = nn.CrossEntropyLoss()
# use the default index = 0 - equivalent to summing all other probabilities
criterion_comp = ComplementCrossEntropyLoss(except_index=0)

from models.distributions import Normal
from models.bayes import NoiseLoss, PriorLoss

# Finally, initialize the ``optimizers''
# Since we keep track of a set of parameters, we also need a set of
# ``optimizers''
if opt.d_optim == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr= 0.5 * 1e-4 , betas=(0.5, 0.999)) # if D_lr = G_lr, it would be unstable
elif opt.d_optim == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr,
                                 momentum=0.9,
                                 nesterov=True,
                                 weight_decay=1e-4)
optimizerGs = []
for netG in netGs:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerGs.append(optimizerG)

# since the log posterior is the average per sample, we also scale down the prior and the noise
gprior_criterion = PriorLoss(prior_std=1., observed=1000.)
gnoise_criterion = NoiseLoss(params=netGs[0].parameters(), scale=math.sqrt(2 * opt.gnoise_alpha / opt.lr),observed=1000.)
dprior_criterion = PriorLoss(prior_std=1., observed=50000.)
dnoise_criterion = NoiseLoss(params=netD.parameters(), scale=math.sqrt(2 * opt.dnoise_alpha * opt.lr), observed=50000.)

# Fixed noise for data generation
fixed_noise = torch.FloatTensor(64, opt.nz).normal_(0, 1).cuda()
fixed_noise = Variable(fixed_noise)

# initialize input variables and use CUDA (optional)
input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    for netG in netGs:
        netG.cuda()
    criterion.cuda()
    criterion_comp.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()


# # fully supervised
# netD_fullsup = _netD(opt.ngpu, num_classes=num_classes)
# netD_fullsup.apply(weights_init)
# criterion_fullsup = nn.CrossEntropyLoss()
# if opt.d_optim == 'adam':
#     optimizerD_fullsup = optim.Adam(netD_fullsup.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# else:
#     optimizerD_fullsup = optim.SGD(netD_fullsup.parameters(), lr=opt.lr,
#                                 momentum=0.9,
#                                 nesterov=True,
#                                 weight_decay=1e-4)
# if opt.cuda:
#     netD_fullsup.cuda()
#     criterion_fullsup.cuda()

# We define a class to calculate the accuracy on test set
# to test the performance of semi-supervised training
def get_test_accuracy(model_d, f, acc, label='semi'):
    # don't forget to do model_d.eval() before doing evaluation
    top1 = AverageMeter()
    for i, (input, target) in enumerate(dataloader_test):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        output = model_d(input_var.view(-1, opt.imageSize))

        probs = output.data[:, 1:]  # discard the zeroth index

        prec1 = accuracy(probs, target, topk=(1,))[0]
        top1.update(prec1[0], input.size(0))
        if i % 50 == 0:
            print("{} Test: [{}/{}]\t Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(label, i, len(dataloader_test), top1=top1))
    acc.append(top1.avg)
    f.write("%s\n" % top1.avg.item())
    print('{label} Test Prec@1 {top1.avg:.2f}'.format(label=label, top1=top1))


acc = []
iteration = 0
f = open('acc.txt', 'w')
for epoch in range(opt.niter):
    top1 = AverageMeter()
    top1_weakD = AverageMeter()
    for i, data in enumerate(dataloader):
        iteration += 1

        # 4. Generator
        for netG in netGs:
            netG.zero_grad()
        fakes = []
        for _idxz in range(opt.numz):
            noise.resize_(opt.batchSize, opt.nz).normal_(0, 1)
            noisev = Variable(noise)
            for _idxm in range(opt.num_mcmc):
                idx = _idxz * opt.num_mcmc + _idxm
                netG = netGs[idx]
                _fake = netG(noisev)
                fakes.append(_fake)
        fake = torch.cat(fakes)
        labelv = Variable(torch.FloatTensor(fake.data.shape[0]).cuda().fill_(real_label))
        output = netD(fake)
        errG = criterion_comp(output)
        if opt.bayes:
            for netG in netGs:
                errG += gprior_criterion(netG.parameters())
                errG += gnoise_criterion(netG.parameters())
        errG.backward()
        D_G_z2 = 1 - torch.nn.functional.softmax(output).data[:, 0].mean()
        for optimizerG in optimizerGs:
            optimizerG.step()
        #######
        # 1. real input
        netD.zero_grad()
        _input, _ = data
        batch_size = _input.size(0)
        if opt.cuda:
            _input = _input.cuda()
        input.resize_as_(_input).copy_(_input)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input).view(-1, opt.imageSize)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion_comp(output)
        errD_real.backward()
        # calculate D_x, the probability that real data are classified
        D_x = 1 - torch.nn.functional.softmax(output).data[:, 0].mean()

        #######
        # 2. Generated input
        fakes = []
        for _idxz in range(opt.numz):
            noise.resize_(batch_size, opt.nz).normal_(0, 1)
            noisev = Variable(noise)
            for _idxm in range(opt.num_mcmc):
                idx = _idxz * opt.num_mcmc + _idxm
                netG = netGs[idx]
                _fake = netG(noisev)
                fakes.append(_fake)
        fake = torch.cat(fakes)
        output = netD(fake.detach())
        # print("output", output.size())
        labelv = Variable(torch.LongTensor(fake.data.shape[0]).cuda().fill_(fake_label))
        # print("labelv", labelv.size())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = 1 - torch.nn.functional.softmax(output).data[:, 0].mean()

        #######
        # 3. Labeled Data Part (for semi-supervised learning)
        for ii, (input_sup, target_sup) in enumerate(dataloader_semi):
            # print("input", input_sup.data.mean())  #suffle, different every time
            # print("target", target_sup.data)
            input_sup, target_sup = input_sup.cuda(), target_sup.cuda()
            break

        input_sup_v = Variable(input_sup.cuda()).view(-1, opt.imageSize)
        # convert target indicies from 0 to 9 to 1 to 10
        target_sup_v = Variable((target_sup + 1).cuda())
        # print("target_sup", target_sup_v)
        output_sup = netD(input_sup_v)
        # print("output_sup", output_sup.data)
        err_sup = criterion(output_sup, target_sup_v)
        err_sup.backward()
        prec1 = accuracy(output_sup.data, target_sup + 1, topk=(1,))[0]
        top1.update(prec1.item(), input_sup.size(0))
        if opt.bayes:
            errD_prior = dprior_criterion(netD.parameters())
            errD_prior.backward()
            errD_noise = dnoise_criterion(netD.parameters())
            errD_noise.backward()
            errD = errD_real + errD_fake + err_sup + errD_prior + errD_noise
        else:
            errD = errD_real + errD_fake + err_sup
        optimizerD.step()




        # 6. get test accuracy after every interval
        if iteration % len(dataloader) == 0:
            # get test accuracy on train and test
            netD.eval()
            get_test_accuracy(netD, f, acc , label='semi')
            # get_test_accuracy(netD_fullsup, iteration, label='sup')
            netD.train()

        # 7. Report for this iteration
            cur_val, ave_val = top1.val, top1.avg
            print('[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f D(x): %.2f D(G(z)): %.2f / %.2f | Acc %.1f / %.1f' % (
            epoch+1, opt.niter, i, len(dataloader), errD.data.item(), errG.data.item(), D_x, D_G_z1, D_G_z2, cur_val, ave_val))

    # after each epoch, save images
    vutils.save_image(_input.view(-1, 1, 28, 28), '%s/real_samples.png' % opt.outf, normalize=False)
    for _zid in range(opt.numz):
        for _mid in range(opt.num_mcmc):
            idx = _zid * opt.num_mcmc + _mid
            netG = netGs[idx]
            netG.eval()
            fake = netG(fixed_noise)
            if not os.path.exists(opt.outf + '/epoch_%d'%(epoch+1)):
                os.makedirs(opt.outf + '/epoch_%d'%(epoch+1))
            vutils.save_image(fake.view(-1, 1, 28, 28).data,
                              '%s/fake_samples_G_z%02d_m%02d.png' % (opt.outf + '/epoch_%d'% (epoch+1), _zid, _mid),
                              normalize= False)
            # for ii, netG in enumerate(netGs):
            #     torch.save(netG.state_dict(), '%s/netG%d_epoch_%d.pth' % (opt.outf, ii, epoch))
            # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
            # torch.save(netD_fullsup.state_dict(), '%s/netD_fullsup_epoch_%d.pth' % (opt.outf, epoch))

xmajorLocator = MultipleLocator(5)
xmajorFormatter = FormatStrFormatter('%d')

plt.figure()
ax = plt.gca()
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
plt.plot(acc)
plt.xlabel('epochs')
plt.ylabel('testing accuracy (%)')
plt.show()