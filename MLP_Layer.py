import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
c = 1e-8


def log_gaussian(x, mu, sigma):
    return (float(-np.log(np.sqrt(2 * np.pi))) - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)).sum()


sigma_1 = torch.cuda.FloatTensor([np.exp(-0)])
sigma_2 = torch.cuda.FloatTensor([np.exp(-6)])
weight = 0.5

def log_mixture_gaussian(x, mu_1, sigma_1, mu_2, sigma_2, weight):
    normal_1 = torch.distributions.normal.Normal(mu_1, sigma_1)
    normal_2 = torch.distributions.normal.Normal(mu_2, sigma_2)
    prob_1 = torch.exp(normal_1.log_prob(x))
    prob_2 = torch.exp(normal_2.log_prob(x))
    return (torch.log(weight * prob_1 + (1 - weight) * prob_2)).sum()

class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super(MLPLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior
        # self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        # self.W_rho = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        # self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        # self.b_rho = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).uniform_(-0.2, 0.2))
        self.W_rho = nn.Parameter(torch.Tensor(n_input, n_output).uniform_(-5, -4))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.2, 0.2))
        self.b_rho = nn.Parameter(torch.Tensor(n_output).uniform_(-5, -4))
        self.lpw = 0
        self.lqw = 0

    def forward(self, X ):
        epsilon_W, epsilon_b = self.get_random()

        W = self.W_mu + torch.log1p(torch.exp(self.W_rho)) * epsilon_W
        b = self.b_mu + torch.log1p(torch.exp(self.b_rho)) * epsilon_b
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)

        # self.lpw = log_gaussian(W, 0, self.sigma_prior) + log_gaussian(b, 0, self.sigma_prior)
        self.lpw = log_mixture_gaussian(W, 0, sigma_1, 0, sigma_2, weight) + \
                   log_mixture_gaussian(b, 0, sigma_1, 0, sigma_2, weight)
        self.lqw = log_gaussian(W, self.W_mu, torch.log1p(torch.exp(self.W_rho))) + \
                   log_gaussian(b, self.b_mu, torch.log1p(torch.exp(self.b_rho)))
        #print("lpw",self.lpw)
        #print("lqw",self.lqw)
        return output

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, 1).to(device)), \
               Variable(torch.Tensor(self.n_output).normal_(0, 1).to(device))