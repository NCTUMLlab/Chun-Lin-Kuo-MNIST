import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from MLP_Layer import MLPLayer
from torch.autograd import Variable
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'vbgan_tan_mse_gd'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST VAE_d_v2')
parser.add_argument('-batch_size', type=int, default= 100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default= 100, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default= 5 * 1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default= 128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default= 32, help='hidden dimension of z (default: 8)')
parser.add_argument('-sigma_prior',type=float, default = torch.tensor(np.exp(-3)).to(device))
parser.add_argument('-n_mc', type=int, default = 5)
parser.add_argument('-n_input', type=int , default= 784)

args = parser.parse_args()


# MNIST dataset
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

c = 1e-8


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.input = args.n_input

        self.enc1 = MLPLayer(self.input, self.dim_h * 4, args.sigma_prior)
        self.bn1 = nn.BatchNorm1d(self.dim_h * 4)
        self.enc1_act = nn.ReLU()
        self.enc2 = MLPLayer(self.dim_h * 4, self.dim_h * 2, args.sigma_prior)
        self.bn2 = nn.BatchNorm1d(self.dim_h * 2)
        self.enc2_act = nn.ReLU()
        self.enc3 = MLPLayer(self.dim_h * 2, self.dim_h, args.sigma_prior)
        self.bn3 = nn.BatchNorm1d(self.dim_h)
        self.enc3_act = nn.ReLU()
        self.enc4 = MLPLayer(self.dim_h, self.n_z, args.sigma_prior)
        self.enc5 = MLPLayer(self.dim_h, self.n_z, args.sigma_prior)

    def encode(self, x):
        h = self.enc1_act(self.bn1(self.enc1(x)))
        h = self.enc2_act(self.bn2(self.enc2(h)))
        h = self.enc3_act(self.bn3(self.enc3(h)))
        return self.enc4(h), self.enc5(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  z, mu, log_var

    def get_lpw_lqw(self):
        lpw = self.enc1.lpw + self.enc2.lpw + self.enc3.lpw + self.enc4.lpw + self.enc5.lpw
        lqw = self.enc1.lqw + self.enc2.lqw + self.enc3.lqw + self.enc4.lqw + self.enc5.lqw
        return lpw, lqw

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.output = args.n_input

        self.dec1 = MLPLayer(self.n_z, self.dim_h, args.sigma_prior)
        self.bn1 = nn.BatchNorm1d(self.dim_h)
        self.dec1_act = nn.ReLU()
        self.dec2 = MLPLayer(self.dim_h, self.dim_h * 2, args.sigma_prior)
        self.bn2 = nn.BatchNorm1d(self.dim_h * 2)
        self.dec2_act = nn.ReLU()
        self.dec3 = MLPLayer(self.dim_h * 2, self.dim_h * 4, args.sigma_prior)
        self.bn3 = nn.BatchNorm1d(self.dim_h * 4)
        self.dec3_act = nn.ReLU()
        self.dec4 = MLPLayer(self.dim_h * 4, self.output, args.sigma_prior)
        #self.bn4 = nn.BatchNorm1d(self.output)
        self.dec4_act = nn.Tanh()

    def decode(self, z):
        h = self.dec1_act(self.bn1(self.dec1(z)))
        h = self.dec2_act(self.bn2(self.dec2(h)))
        h = self.dec3_act(self.bn3(self.dec3(h)))
        return self.dec4_act((self.dec4(h)))

    def forward(self, z):
        x = self.decode(z)
        return x

    def get_lpw_lqw(self):
        lpw = self.dec1.lpw + self.dec2.lpw + self.dec3.lpw + self.dec4.lpw
        lqw = self.dec1.lqw + self.dec2.lqw + self.dec3.lqw + self.dec4.lqw
        return lpw, lqw

def forward_pass_samples(x, real_labels):
    enc_kl, dec_kl, rec_scores , sam_scores = torch.zeros(args.n_mc), torch.zeros(args.n_mc) ,torch.zeros(args.n_mc), torch.zeros(args.n_mc)
    enc_log_likelihoods, dec_log_likelihoods = torch.zeros(args.n_mc), torch.zeros(args.n_mc)
    for i in range(args.n_mc):
        z = torch.randn(args.batch_size, args.n_z).to(device)  # z~N(0,1)
        z_enc, mu, log_var = encoder(x)
        x_rec = decoder(z_enc)
        x_sam = decoder(z)
        #assert ((x_rec >= 0.) & (x_rec <= 1.)).all()
        #reconst_loss = F.binary_cross_entropy(x_rec, x , reduction = 'sum')
        reconst_loss = mse(x_rec, x)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # kl p(z) between q(z|x)

        outputs_rec = discriminator(x_rec)
        syn_rec_loss = bcewl_sum(outputs_rec, real_labels) #hope reconstruction could be sharper
        outputs_sam = discriminator(x_sam)  #hope prioir sample z could be generated better
        syn_sam_loss = bcewl_sum(outputs_sam, real_labels)

        #syn_loss = (-torch.log(fake_score) + torch.log(1-fake_score)).mean()
        # print("rec",reconst_loss.item())
        # print("kl",kl_div)
        # print("syn_rec",(syn_rec_loss).mean().item())
        # print("syn_sam",syn_sam_loss.mean().item())
        enc_log_pw, enc_log_qw = encoder.get_lpw_lqw()
        dec_log_pw, dec_log_qw = decoder.get_lpw_lqw()
        enc_log_likelihood = reconst_loss + kl_div
        dec_log_likelihood = reconst_loss + ( syn_rec_loss + syn_sam_loss ) *10

        enc_kl[i] = enc_log_qw - enc_log_pw
        dec_kl[i] = dec_log_qw - dec_log_pw
        enc_log_likelihoods[i] = enc_log_likelihood
        dec_log_likelihoods[i] = dec_log_likelihood
        rec_scores[i] = outputs_rec.mean()
        sam_scores[i] = outputs_sam.mean()

    return enc_kl.mean(), dec_kl.mean(), enc_log_likelihoods.mean(), dec_log_likelihoods.mean(), rec_scores.mean(), sam_scores.mean()

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.input = args.n_input

        self.main = nn.Sequential(
            nn.Linear(self.input, self.dim_h * 2),
            #nn.BatchNorm1d(self.dim_h * 2),  #should try again
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h * 2, self.dim_h * 2),
            nn.BatchNorm1d(self.dim_h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h * 2, 1),
            #nn.Sigmoid()
        )
    def forward(self,x):
        outputs = self.main(x)
        return outputs

def reset_grad():
    dis_optimizer.zero_grad()
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

def criterion(kl, log_likelihood):
    return   kl / len(train_loader) + log_likelihood

def criterion_reW(kl, i, log_likelihood):
    M = len(train_loader)
    weight = (2^(M - i)) / (2^M -1)
    # print("kl", ((kl * weight) / M).item())
    # print("loglikelihood", log_likelihood.item())
    return   (kl * weight) / M + log_likelihood

encoder = Encoder(args).to(device)
decoder = Decoder(args).to(device)
discriminator = Discriminator(args).to(device)
enc_optimizer = torch.optim.Adam(encoder.parameters(), lr = args.lr)
dec_optimizer = torch.optim.Adam(decoder.parameters(), lr = args.lr)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.1 * args.lr)

bcewl_sum = nn.BCEWithLogitsLoss(reduction = 'sum')
bcewl = nn.BCEWithLogitsLoss()
bce = nn.BCELoss(reduction = 'sum')
mse = nn.MSELoss(reduction = 'sum')
# dis_scheduler = StepLR(dis_optimizer, step_size=10, gamma=0.1)
# enc_scheduler = StepLR(enc_optimizer, step_size=10, gamma=0.1)
# dec_scheduler = StepLR(dec_optimizer, step_size=10, gamma=0.1)

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


# Start training
encoder.train()
decoder.train()
discriminator.train()

for epoch in range(args.epochs):
    # dis_scheduler.step(epoch=10)
    # dec_scheduler.step(epoch=10)
    # enc_scheduler.step(epoch=10)
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device).view(-1, args.n_input)
        z = torch.randn(args.batch_size, args.n_z).to(device)  # z~N(0,1)
        real_labels = torch.ones(args.batch_size, 1).to(device)
        fake_labels = torch.zeros(args.batch_size, 1).to(device)
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        free_params(decoder)
        free_params(encoder)
        frozen_params(discriminator)
        enc_kl, dec_kl, enc_log_likelihood, dec_log_likelihood, rec_scores, sam_scores = forward_pass_samples(x, real_labels)
        enc_loss = criterion_reW(enc_kl, i, enc_log_likelihood)
        dec_loss = criterion_reW(dec_kl, i, dec_log_likelihood)

        reset_grad()
        enc_loss.backward(retain_graph=True)
        enc_optimizer.step()

        reset_grad()
        dec_loss.backward(retain_graph=True)
        dec_optimizer.step()

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        frozen_params(decoder)
        frozen_params(encoder)
        free_params(discriminator)
        outputs = discriminator(x)
        d_loss_real = bcewl(outputs, real_labels)
        real_score = outputs


        x_sam = decoder(z)
        outputs_sam = discriminator(x_sam)
        d_loss_fake = bcewl(outputs_sam, fake_labels)

        z_enc, _, _ = encoder(x)
        x_rec = decoder(z_enc)
        outputs_rec = discriminator(x_rec)
        d_loss_rec = bcewl(outputs_rec, fake_labels)


        d_loss = d_loss_real + (d_loss_fake + d_loss_rec) * 0.5 # maybe * 0.5

        reset_grad()
        d_loss.backward()
        dis_optimizer.step()
        # print("fake",fake_score.mean().item())
        # print("real",real_score.mean().item())


        if (i + 1) % 600 == 0:
            print("Epoch[{}/{}], Step [{}/{}], enc_Loss: {:.4f} ,dec_Loss: {:.4f}, d_Loss: {:.4f}, rec_socre: {:.4f}, sam_socre: {:.4f},"
                  .format(epoch + 1, args.epochs, i + 1, len(train_loader), enc_loss.item(), dec_loss.item(), d_loss.item() ,rec_scores.item(), sam_scores.item()))

    with torch.no_grad():
        if (epoch + 1) % 1 == 0:
            encoder.eval()
            decoder.eval()
            # Save the sampled images
            z = torch.randn(args.batch_size, args.n_z).to(device)
            x_sam = decoder(z).view(-1, 1, 28, 28)
            save_image(x_sam, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

            test_iter = iter(test_loader)
            test_data = next(test_iter)



            # Save the reconstructed images
            z_enc, _, _ = encoder(Variable(test_data[0].view(-1, args.n_input)).to(device))
            x_rec = decoder(z_enc).to(device).view(args.batch_size, 1, 28, 28)
            x_concat = torch.cat([test_data[0].view(-1, 1, 28, 28).to(device), x_rec.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))

# Save the model checkpoints
torch.save(encoder.state_dict(), './' + sample_dir + '/encoder.ckpt')
torch.save(decoder.state_dict(), './' + sample_dir + '/decoder.ckpt')
torch.save(discriminator.state_dict(), './' + sample_dir + '/discriminator.ckpt')