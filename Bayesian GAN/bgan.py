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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from statsutil import AverageMeter, accuracy
from tensorboard_logger import configure, log_value
import argparse
from torchvision.datasets import MNIST

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--imageSize', type=int, default=784)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default= 20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=128, help='number of GPUs to use')
parser.add_argument('--outf', default='modelfiles/MNIST_dense', help='folder to output images and model checkpoints')
parser.add_argument('--numz', type=int, default=1, help='The number of set of z to marginalize over.')
parser.add_argument('--num_mcmc', type=int, default=10, help='The number of MCMC chains to run in parallel')
parser.add_argument('--num_semi', type=int, default=400, help='The number of semi-supervised samples')
parser.add_argument('--gnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--dnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--d_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--g_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--stats_interval', type=int, default=10, help='Calculate test accuracy every interval')
parser.add_argument('--tensorboard', type=int, default=1, help='')
parser.add_argument('--bayes', type=int, default=1, help='Do Bayesian GAN or normal GAN')
import sys; sys.argv=['']; del sys
opt = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if opt.tensorboard:
    configure(opt.outf)


dataset = MNIST(root='./data/',
                 train=True,
                 transform=transforms.ToTensor(),
                 download=True)

dataset_test = MNIST(root='./data/',
                train=False,
                transform = transforms.ToTensor(),
                download=True)

dataloader = torch.utils.data.DataLoader(dataset=dataset,
                        batch_size = opt.batchSize,
                        shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                         batch_size = opt.batchSize,
                         shuffle=False)

from partial_dataset import PartialDataset
# partial dataset for semi-supervised training
dataset_partial = PartialDataset(dataset, opt.num_semi)

dataloader_semi = torch.utils.data.DataLoader(dataset_partial, batch_size = opt.batchSize,
                                    shuffle=True, num_workers=1)


from models.discriminators import dense_D
from models.generators import dense_G
from statsutil import weights_init

netGs = []
for _idxz in range(opt.numz):
    for _idxm in range(opt.num_mcmc):
        netG = dense_G(opt.ngpu, opt.nz).to(device)
        netG.apply(weights_init)
        netGs.append(netG)

num_classes = 11
netD = dense_D(opt.ngpu, num_classes= num_classes).to(device)

from ComplementCrossEntropyLoss import ComplementCrossEntropyLoss
criterion = nn.CrossEntropyLoss()
criterion_comp = ComplementCrossEntropyLoss(except_index=0)

from models.distributions import Normal
from models.bayes import NoiseLoss, PriorLoss

if opt.d_optim == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
elif opt.d_optim == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr, momentum=0.9, esterov=True, weight_decay=1e-4)

optimizerGs = []
for netG in netGs:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerGs.append(optimizerG)

gprior_criterion = PriorLoss(prior_std=1., observed=1000.)
gnoise_criterion = NoiseLoss(params=netGs[0].parameters(), scale=math.sqrt(2*opt.gnoise_alpha/opt.lr), observed=1000.)
dprior_criterion = PriorLoss(prior_std=1., observed=50000.)
dnoise_criterion = NoiseLoss(params=netD.parameters(), scale=math.sqrt(2*opt.dnoise_alpha*opt.lr), observed=50000.)

# Fixed noise for data generation
fixed_noise = torch.randn(opt.batchSize, opt.nz).to(device)

# initialize input variables and use CUDA (optional)
input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize).to(device)
noise = torch.FloatTensor(opt.batchSize, 1, opt.nz).to(device)
label = torch.FloatTensor(opt.batchSize).to(device)
# real_labels = torch.ones(opt.batchSize, 1).to(device)
# fake_labels = torch.zeros(opt.batchSize, 1).to(device)

#We define a class to calculate the accuracy on test set
# to test the performance of semi-supervised training
def get_test_accuracy(model_d,f, iteration, label='semi'):
    # don't forget to do model_d.eval() before doing evaluation
    top1 = AverageMeter()
    for i, (input, target) in enumerate(dataloader_test):
        target = target.to(device)
        input = input.view(-1, opt.imageSize).to(device)
        output = model_d(input)

        probs = output.data[:, 1:] # discard the zeroth index
        prec1 = accuracy(probs, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))
        if i % 50 == 0:
              print("{} Test: [{}/{}]\t Prec@1 {top1.val:.3f} ({top1.avg:.3f})"\
                .format(label, i, len(dataloader_test), top1=top1))
    print('{label} Test Prec@1 {top1.avg:.2f}'.format(label=label, top1=top1))
    f.write("%s\n" % top1.avg)
    log_value('test_acc_{}'.format(label), top1.avg, iteration)


iteration = 0
f = open('acc.txt', 'w')
for epoch in range(opt.niter):
    top1 = AverageMeter()
    top1_weakD = AverageMeter()
    for i, (x, _) in enumerate(dataloader):
        iteration += 1
        x = x.view(-1, opt.imageSize).to(device)
        # 1. real input
        netD.zero_grad()

        output = netD(x)
        errD_real = criterion_comp(output)
        errD_real.backward()
        # calculate D_x, the probability that real data are classified
        D_x = 1 - torch.nn.functional.softmax(output).data[:, 0].mean()

        # 2. Generated input
        fakes = []
        for _idxz in range(opt.numz):
            noise = torch.randn(opt.batchSize, opt.nz).to(device)
            for _idxm in range(opt.num_mcmc):
                idx = _idxz * opt.num_mcmc + _idxm
                netG = netGs[idx]
                _fake = netG(noise)
                fakes.append(_fake)
        fake = torch.cat(fakes)
        output = netD(fake.detach())  #(batchsize * n_mcmc , n_classes)
        labelv = Variable(torch.LongTensor(fake.data.shape[0]).cuda().fill_(0))  #(batchsize * n_mcmc)
        errD_fake = criterion(output, labelv)
        errD_fake.backward()

        D_G_z1 = 1 - torch.nn.functional.softmax(output).data[:, 0].mean()

        # 3. Labeled Data Part (for semi-supervised learning)
        for ii, (input_sup, target_sup) in enumerate(dataloader_semi):
            input_sup = input_sup.view(-1, opt.imageSize).to(device)
            target_sup = (target_sup + 1).to(device)
            break

        # convert target indicies from 0 to 9 to 1 to 10
        output_sup = netD(input_sup)
        err_sup = criterion(output_sup, target_sup)
        err_sup.backward()
        prec1 = accuracy(output_sup.data, target_sup, topk=(1,))[0]
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

        # 4. Generator
        for netG in netGs:
            netG.zero_grad()
        labelv = Variable(torch.FloatTensor(fake.data.shape[0]).cuda().fill_(1))
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

        # 6. get test accuracy after every interval
        if iteration % 938 == 0:
            # get test accuracy on train and test
            netD.eval()
            get_test_accuracy(netD, f, iteration, label='semi')
            netD.train()

    # 7. Report for this iteration
    cur_val, ave_val = top1.val, top1.avg
    log_value('train_acc', top1.avg, iteration)
    print('[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f D(x): %.2f D(G(z)): %.2f / %.2f | Acc %.1f / %.1f'
          % ((epoch + 1), opt.niter, i, len(dataloader),
             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, cur_val, ave_val))
    # after each epoch, save images
    vutils.save_image(x.view(-1, 1, 28, 28), '%s/real_samples.png' % opt.outf)
    if not os.path.exists(opt.outf + '/epoch_%d' % (epoch + 1)):
        os.makedirs(opt.outf + '/epoch_%d' % (epoch + 1))
    for _zid in range(opt.numz):
        for _mid in range(opt.num_mcmc):
            idx = _zid * opt.num_mcmc + _mid
            netG = netGs[idx]
            fake = netG(fixed_noise)
            vutils.save_image(fake.data.view(-1,1,28,28),'%s/fake_samples_G_z%02d_m%02d.png' % (
                              opt.outf + '/epoch_%d' % (epoch + 1), _zid, _mid))

    # for ii, netG in enumerate(netGs):
    #     torch.save(netG.state_dict(), '%s/netG%d_epoch_%d.pth' % (opt.outf, ii, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    # torch.save(netD_fullsup.state_dict(), '%s/netD_fullsup_epoch_%d.pth' % (opt.outf, epoch))
