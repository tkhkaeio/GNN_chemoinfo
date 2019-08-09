import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from numpy import exp, log

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        nn.init.normal_(self.linear1.weight, args.mean, args.std)
        nn.init.zeros_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, args.mean, args.std)
        nn.init.zeros_(self.linear2.bias)
        self.activation=args.activation
        
    def forward(self, x):
        if self.activation=="ReLU":
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
        elif self.activation=="LeakyReLU":
            x = F.leaky_relu(self.linear1(x))
            x = F.leaky_relu(self.linear2(x))

        return x

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, args, activation="ReLU"):
        super(Linear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.linear1.weight, args.mean, args.std)
        #nn.init.ones_(self.linear1.weight) ###
        nn.init.zeros_(self.linear1.bias)
        if activation is None: self.activation = None
        else: self.activation = args.activation

    def forward(self, x):
        if self.activation=="ReLU":
            x = F.relu(self.linear1(x))
        elif self.activation=="LeakyReLU":
            x = F.leaky_relu(self.linear1(x))
        else:
            x = self.linear1(x)
        return x

def Hg_check(Hg, args, phase=None):
    try:
        if(np.sum(Hg.cpu().detach().numpy())==0): raise ValueError("hg calculation error in {} phase\n".format(phase))
    except ValueError as e: 
        print(e)