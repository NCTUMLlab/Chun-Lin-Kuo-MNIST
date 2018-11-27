import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from MLP_Layer import MLPLayer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import numpy as np




#torch.manual_seed(123)
c = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')
parser.add_argument('-batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=256, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=20, help='hidden dimension of z (default: 8)')
parser.add_argument('-LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('-sigma_z', type=float, default= 1, help='variance of hidden dimension (default: 1)')
parser.add_argument('-sigma_prior',type=float, default = torch.tensor(np.exp(-3)).to(device))
parser.add_argument('-n_mc', type=int, default = 5)
parser.add_argument('-n_input', type=int , default= 784)

args = parser.parse_args()

trainset = MNIST(root='./data/',
                 train=True,
                 transform=transforms.ToTensor(),
                 download=True)

testset = MNIST(root='./data/',
                train=False,
                transform=transforms.ToTensor(),
                download=True)

train_loader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=testset,
                         batch_size=args.batch_size,
                         shuffle=False)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.input = args.n_input

        self.l1 = MLPLayer(self.input, self.dim_h , args.sigma_prior)
        self.l1_act = nn.ReLU()
        self.l2 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l2_act = nn.ReLU()
        self.l3 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l3_act = nn.ReLU()
        self.l4 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l4_act = nn.ReLU()
        self.l5 = MLPLayer(self.dim_h, self.n_z, args.sigma_prior)


    def forward(self, x):
        output = self.l1_act(self.l1(x))
        output = self.l2_act(self.l2(output))
        output = self.l3_act(self.l3(output))
        output = self.l4_act(self.l4(output))
        output = self.l5(output)

        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw + self.l4.lpw + self.l5.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw + self.l4.lqw + self.l5.lqw
        return lpw, lqw


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.output = args.n_input
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.l1 = MLPLayer(self.n_z, self.dim_h, args.sigma_prior)
        self.l1_act = nn.ReLU()
        self.l2 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l2_act = nn.ReLU()
        self.l3 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l3_act = nn.ReLU()
        self.l4 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l4_act = nn.ReLU()
        self.l5 = MLPLayer(self.dim_h, self.output, args.sigma_prior)
        self.l5_act = nn.Sigmoid()

    def forward(self, z):
        output = self.l1_act(self.l1(z))
        output = self.l2_act(self.l2(output))
        output = self.l3_act(self.l3(output))
        output = self.l4_act(self.l4(output))
        output = self.l5_act(self.l5(output))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw + self.l4.lpw + self.l5.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw + self.l4.lqw + self.l5.lqw
        return lpw, lqw

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, 1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x

def forward_pass_samples(images, real_labels):
    enc_kl, dec_kl, enc_scores, sam_scores = torch.zeros(args.n_mc), torch.zeros(args.n_mc), torch.zeros(args.n_mc), torch.zeros(args.n_mc)
    enc_log_likelihoods, dec_log_likelihoods = torch.zeros(args.n_mc), torch.zeros(args.n_mc)
    for i in range(args.n_mc):
        z_enc = encoder(images)
        x_rec = decoder(z_enc)
        rec_loss = mse_sum(x_rec, images)
        d_enc = discriminator(z_enc)
        #div_loss =  -args.LAMBDA * (torch.log(d_enc)).sum()
        div_loss = bcewl_sum(d_enc, real_labels)
        #print("rec_loss",rec_loss.item())
        #print("div_loss",div_loss.item())
        enc_log_likelihood = rec_loss + div_loss
        dec_log_likelihood = rec_loss + div_loss

        enc_log_pw, enc_log_qw = encoder.get_lpw_lqw()
        dec_log_pw, dec_log_qw = decoder.get_lpw_lqw()

        enc_kl[i] = enc_log_qw - enc_log_pw
        dec_kl[i] = dec_log_qw - dec_log_pw
        enc_log_likelihoods[i] = enc_log_likelihood
        dec_log_likelihoods[i] = dec_log_likelihood
        enc_scores[i] = d_enc.mean()
        #sam_scores[i] = d_sam.mean()


    return enc_kl.mean(), dec_kl.mean(), enc_log_likelihoods.mean(), dec_log_likelihoods.mean(), enc_scores.mean()#, sam_scores.mean()

encoder, decoder, discriminator = Encoder(args).to(device), Decoder(args).to(device), Discriminator(args).to(device)
mse_sum = nn.MSELoss(reduction = 'sum')
bcewl_sum = nn.BCEWithLogitsLoss(reduction= 'sum')
bcewl = nn.BCEWithLogitsLoss()

def criterion(kl, log_likelihood):
    return   kl / len(train_loader) + log_likelihood

def criterion_reW(kl, i, log_likelihood):
    M = len(train_loader)
    weight = (2^(M - i)) / (2^M -1)
    # print("kl",kl)
    # print("loglikelihood",log_likelihood)
    return   (kl * weight) / M + log_likelihood

encoder.train()
decoder.train()
discriminator.train()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr = args.lr)
dec_optim = optim.Adam(decoder.parameters(), lr = args.lr)
dis_optim = optim.Adam(discriminator.parameters(), lr = 0.1 * args.lr)

# enc_scheduler = StepLR(enc_optim, step_size=60, gamma=0.5)
# dec_scheduler = StepLR(dec_optim, step_size=60, gamma=0.5)
# dis_scheduler = StepLR(dis_optim, step_size=60, gamma=0.5)





for epoch in range(args.epochs):
    step = 0
    #enc_scheduler.step(epoch = 60)
    #dec_scheduler.step(epoch = 60)
    #dis_scheduler.step(epoch = 60)
    #print(enc_scheduler.get_lr())
    for images, _ in tqdm(train_loader):
        images = images.to(device)
        images = images.view(-1,args.n_input)
        real_labels = torch.ones(args.batch_size, 1).to(device)
        fake_labels = torch.zeros(args.batch_size, 1).to(device)

        # ======== Train Generator ======== #
        #print("-------------G-----------------")
        free_params(decoder)
        free_params(encoder)
        frozen_params(discriminator)

        enc_kl, dec_kl, enc_log_likelihood, dec_log_likelihood, enc_scores = forward_pass_samples(images,real_labels)
        # enc_loss = criterion(enc_kl, enc_log_likelihood)
        # dec_loss = criterion(dec_kl, dec_log_likelihood)
        enc_loss = criterion_reW(enc_kl, (step+1), enc_log_likelihood)
        dec_loss = criterion_reW(dec_kl, (step+1), dec_log_likelihood)

        encoder.zero_grad()
        enc_loss.backward(retain_graph = True)
        enc_optim.step()

        decoder.zero_grad()
        dec_loss.backward()
        dec_optim.step()

        # ======== Train Discriminator ======== #
        # print("-------------D-----------------")
        frozen_params(decoder)
        frozen_params(encoder)
        free_params(discriminator)

        z_sam = (torch.randn(args.batch_size, args.n_z) * args.sigma_z).to(device)  # images.size()[0] -> 100
        d_sam = discriminator(z_sam)
        d_loss_real = bcewl(d_sam, real_labels)

        z_enc = encoder(images)
        d_enc = discriminator(z_enc)
        d_loss_fake = bcewl(d_enc, fake_labels)

        # dis_loss =  (-torch.log(d_sam).mean() - torch.log(1 - d_enc).mean())
        # dis_loss = args.LAMBDA * (-torch.log(d_sam).mean() + torch.log(d_enc + c).mean())
        dis_loss = d_loss_real + d_loss_fake
        # print("dis_loss",dis_loss)
        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()
        dis_loss.backward()
        dis_optim.step()
        # dis_scheduler.step()
        step += 1

        if (step + 1) % len(train_loader) == 0:
            print("Epoch[{}/{}], Step [{}/{}], enc_Loss: {:.4f} ,dec_Loss: {:.4f}, dis_Loss: {:.4f}, enc_socre: {:.4f}"
                  .format(epoch + 1, args.epochs, step + 1, len(train_loader), enc_log_likelihood.item(), dec_log_likelihood.item(), dis_loss.item() , enc_scores.item()))

    with torch.no_grad():
        if (epoch + 1) % 1 == 0:
            encoder.eval()
            decoder.eval()
            if not os.path.isdir('./data/sample_images'):
                os.makedirs('data/sample_images')
            z = (torch.randn(args.batch_size, args.n_z)* args.sigma_z).to(device)
            x_sam = decoder(z).view(-1, 1, 28, 28)
            save_image(x_sam,'./data/sample_images/wae_gan_sam_sch_%d.png' % (epoch + 1))

            test_iter = iter(test_loader)
            test_data = next(test_iter)

            z_enc = encoder(Variable(test_data[0].view(-1, args.n_input)).to(device))
            x_rec = decoder(z_enc).to(device).view(args.batch_size, 1, 28, 28)

            if not os.path.isdir('./data/reconst_images'):
                os.makedirs('data/reconst_images')

            x_concat = torch.cat([test_data[0].view(-1, 1, 28, 28).to(device), x_rec.data.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat,'./data/reconst_images/wae_gan_rec_sch_%d.png' % (epoch + 1))
