import torch.nn as nn
import torch
import numpy as np

eps = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)

class Subnet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, D, act_fn):
        super(Subnet, self).__init__()
        # state_size -> hidden_size -> num_asset
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.act_fn = act_fn
        self.l = D[0] # lower bound
        self.u = D[1] # upper bound

        self.hidden_layer1 = nn.Linear(input_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        activation = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }[self.act_fn]
        x = x.view(-1, self.input_size)
        x = activation(self.hidden_layer1(x))
        x = activation(self.hidden_layer2(x))
        x = nn.Tanh()(self.output_layer(x))

        b = (self.u - self.l) / 2
        a = (self.u + self.l) / (self.u - self.l)
        out = (x + a) * b # the range of out => D ???
        return out


class Fullnet(nn.Module):

    def __init__(self, spot_price_0, strike_price, state_size, num_asset, num_step, r_f, u_gamma, D=[0, 1],  asset_model='BS', hidden_size=50, act_fn='relu'):
        super(Fullnet, self).__init__()
        self.state_size = state_size
        self.num_asset = num_asset
        self.num_step = num_step
        self.r_f = r_f
        self.u_gamma = u_gamma # hyper-para [0, 1]
        self.D = D # range of output
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.asset_model = asset_model
        self.spot_price_0 = spot_price_0
        self.strike_price = strike_price # strike_price

        # define num_step subnets
        self.subnets = nn.ModuleList([Subnet(state_size, num_asset, hidden_size, D, act_fn) for k in range(num_step)])

    def forward(self, x):

        if self.asset_model == 'BS':   # state_variable = W
            W_k = torch.ones([x.shape[0], 1], device=device, requires_grad=True) # init_wealth = 1
            x.requires_grad = False
            spot_price_k = self.spot_price_0  # initialize
            #print('spot_price_0 in model.py', self.spot_price_0)
            for k in range(self.num_step): # n * K * p
                r_k = x[:, k, :].clone().detach()
                #print('r_k in model.py', r_k)
                spot_price_k = spot_price_k * (1 + x[:, k, :])
                #print('spot_price_k in model.py', spot_price_k)

                s_k = W_k
                g_k = self.subnets[k](s_k) # state_variable = W
                W_k = W_k * (torch.sum(r_k * g_k, dim=1).view(-1,1) + self.r_f)
        else:
            print("Please set asset_model == 'BS'!!!")
        #elif self.asset_model == 'AR': # state_variable = (W, R_K)

            #W_k = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
            #x.requires_grad = False

 #          #wealth = torch.ones(x.shape[0], x.shape[1])

            #for k in range(1, self.num_step+1):
            #    r_k_previous = x[:, k-1, :].clone().detach()
            #    s_k = torch.cat((W_k / self.u_gamma, r_k_previous), dim=1)  ## W_k normalization 했다 did
            #    g_k = self.subnets[k-1](s_k)
            #    r_k = x[:, k, :].clone().detach()
            #    W_k = W_k * (torch.sum(r_k * g_k, dim=1).view(-1, 1) + self.r_f) # update W_k
            #    wealth[:, k] = W_k.view(-1).clone().detach()

        #utility = torch.pow((W_k - self.u_gamma / 2), 2) # utility func, to be changed

        h = torch.max(torch.sum(spot_price_k, dim=1) - self.strike_price, torch.zeros_like(torch.sum(spot_price_k, dim=1) - self.strike_price))
        h = h.view([h.shape[0], 1])
        y = W_k - h

        utility = torch.pow(1 + self.u_gamma * torch.sign(y), 2) * torch.pow(y, 2) / 2  # (1 - \gamma * sign(x))**2 * \frac{y**2}{2}
        return utility







