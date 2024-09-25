## import packages
import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
from generator import *
from optimizers import THEOPOULA, TUSLA, SGLD, SGHMC
from utils import ReturnDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from models import Fullnet
import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## parsing
parser = argparse.ArgumentParser('Hedging') # ('portfolio selection')
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--epochs', default=200, type=int, help='# of epochs')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--strike_price', default=5, type=float)
parser.add_argument('--spot_price', default=1, type=float)
parser.add_argument('--act_fn', default='relu', type=str)
parser.add_argument('--hidden_size', default=5, type=int, help='number of neurons') ###
parser.add_argument('--optimizer', default='adam', type=str) ###
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--eta', default=0, type=float)
parser.add_argument('--beta', default='1e12', type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--eps', default=1e-1, type=float)
parser.add_argument('--TUSLA_r', default=0.5, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--patience', default=5, type=int)
parser.add_argument('--lr_gamma', default=0.1, type=float) ###
parser.add_argument('--steplr', default=50, type=int)
parser.add_argument('--T_max', default=100, type=int)
parser.add_argument('--eta_min', default=0.01, type=float)
parser.add_argument('--log_dir', default='./logs/', type=str)
parser.add_argument('--ckpt_dir', default='./ckpt/', type=str)
parser.add_argument('--when', nargs="+", type=int, default=[-1])

# model parameters
#parser.add_argument('--asset_model', default='BS', choices=['AR', 'CCC-GARCH', 'BS'], type=str)
parser.add_argument('--asset_model', default='BS', choices=['BS'], type=str)
parser.add_argument('--num_asset', default=5, type=int)
parser.add_argument('--m', default=5, type=int) # m, num_asset, int
parser.add_argument('--num_path', default=20000, type=int)
parser.add_argument('--R_f', default=1.03, type=float)
parser.add_argument('--u_gamma', default=0.5, type=float, help='parameter for the utility function, (0, 1)')
parser.add_argument('--num_step', default=40, type=int)
parser.add_argument('--scheduler_type', default='step', type=str)
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--beta_annealing', action='store_true')
parser.add_argument('--beta_gamma', default=10, type=float)
parser.add_argument('--beta_step', default=25, type=int)


args = parser.parse_args()
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
start = time.time()
# set the bounding constraints
if args.asset_model == 'BS':
    if args.num_asset == 5:
        D = [0, 1.5] # bounds
        time_step = 1/args.num_step # 1 / 40
    elif args.num_asset == 50:
        D = [0, 1]
        time_step = 1/args.num_step # 1 / 40
    elif args.num_asset == 100:
        D = [0, 1]
        time_step = 1/args.num_step # 1 / 30
    else:
        print('no explicit bounding constraints ==> set [0, 1]^p')
        D = [0, 1]

    spot_price_0 = args.spot_price
    strike_price = args.strike_price # strike_price
    state_size = 1
    m = args.m
    num_step = int(1 / time_step)
    print(time_step)
    R_f = np.exp(0.03 * time_step)


else:
    print("Please set asset_model=='BS'!!!")

num_batch = int(np.ceil(args.num_path / args.batch_size))


def get_ckpt_name(model='BS', num_asset=5, seed=111, optimizer='sgld', lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=5e-4, lr_gamma=0.1, steplr=100,
                  beta=1e12, gamma=0.5, scheduler=False, hidden_size=5, T_max=100, eta_min=0.001, num_step=10,
                  batch_size=64, scheduler_type='step', epochs=100, patience=5,
                  beta_annealing=False, beta_gamma=10, beta_step=25, r=0.5, num_path=1000):
    name = {
        'sghmc': 'seed{}-lr{}-beta{:.1e}-gamma{}-wdecay{}'.format(seed, lr, beta, gamma, weight_decay),
        'sgld': 'seed{}-lr{}-beta{:.1e}-wdecay{}'.format(seed, lr, beta, weight_decay),
        'adam': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'amsgrad': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'theopoula': 'seed{}-lr{}-eps{}-wdecay{}-beta{:.1e}'.format(seed, lr, eps, weight_decay, beta),
        'tusla': 'seed{}-lr{}-r{}-wdecay{}-beta{:.1e}'.format(seed, lr, r, weight_decay, beta),
        'sgd': 'seed{}-lr{}'.format(seed, lr),
        'rmsprop': 'seed{}-lr{}'.format(seed, lr)
    }[optimizer]
    if scheduler:
        name = name + '-scheduler{}-steplr{}-lrgamma{}'.format(scheduler_type, steplr, lr_gamma)

    if beta_annealing:
        name = name + '-beta_annealing{}-betagamma{}-betastep{}'.format(beta_annealing, beta_gamma, beta_step)

    return '{}-p{}-m{}-strike{}-num_step{}-hs{}-bs{}-{}-{}-epochs{}-n_paths{}'.format(model, num_asset, m , strike_price, num_step, hidden_size, batch_size, optimizer, name, epochs, num_path)

save = get_ckpt_name(model=args.asset_model, seed=args.seed, num_asset=args.num_asset, optimizer=args.optimizer, lr=args.lr,
                     eps=args.eps, num_step=num_step, weight_decay=args.weight_decay, lr_gamma=args.lr_gamma, steplr=args.steplr,
                     beta=args.beta, gamma=args.gamma, scheduler=args.scheduler, hidden_size=args.hidden_size, T_max=args.T_max, eta_min=args.eta_min,
                     batch_size=args.batch_size, scheduler_type=args.scheduler_type, epochs=args.epochs, patience=args.patience,
                     beta_annealing=args.beta_annealing, beta_gamma=args.beta_gamma, beta_step=args.beta_step, momentum=args.momentum, r=args.TUSLA_r,
                     num_path = args.num_path
                     )


## Building model

print('\n=> Building model.. on {%s}'%device)

net = Fullnet(spot_price_0=spot_price_0,
              strike_price=strike_price,
              state_size=state_size,
              num_asset=args.num_asset,
              num_step=num_step,
              r_f=R_f,
              u_gamma=args.u_gamma,
              D=D,
              asset_model=args.asset_model,
              hidden_size=args.hidden_size,
              act_fn=args.act_fn)
net.to(device)

print('\n==> Setting optimizer.. use {%s}'%args.optimizer)
print('setting: {}'.format(save))


optimizer = { 'sghmc': SGHMC(net.parameters(), lr=args.lr, beta=args.beta, gamma=args.gamma, weight_decay=args.weight_decay),
              'sgld': SGLD(net.parameters(), lr=args.lr, beta=args.beta, weight_decay=args.weight_decay),
              'adam': optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
              'amsgrad': optim.Adam(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay),
              'theopoula': THEOPOULA(net.parameters(), lr=args.lr, eta=args.eta, beta=args.beta, eps=args.eps, weight_decay=args.weight_decay),
              'tusla': TUSLA(net.parameters(), lr=args.lr, r=args.TUSLA_r, beta=args.beta, weight_decay=args.weight_decay),
              'sgd': optim.SGD(net.parameters(), lr=args.lr),
              'rmsprop': optim.RMSprop(net.parameters(), lr=args.lr),
}[args.optimizer]




## Training
print('\n==> Start training ')

history = {'train_score': [],
           'test_score': [],
           'running_time': 0,
           'training_time': 0,
           'best_epoch': 0,
           }
#state = {}
best_score = 99999
training_time1 = 0

train_data = generate_path(args.asset_model, args.num_asset, args.m, args.num_path, time_step, R_f)


print(args.num_step, time_step, train_data.shape, np.mean(train_data), np.max(train_data), np.min(train_data), np.std(train_data), R_f)

def adjust_learning_rate(optimizer, epoch, steplr=50, lr_gamma=0.1):
    if epoch % steplr == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_gamma

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, min_lr=args.lr*0.001)

def train(epoch, net):
    global training_time1
    print('\n Epoch: %d'%epoch)
    net.train()
    train_score = 0

    train_data = generate_path(args.asset_model, args.num_asset, args.m, args.num_path, time_step, R_f)

    start_time = time.time()

    for batch_idx in range(num_batch):
        start = batch_idx * args.batch_size
        end = np.minimum((batch_idx + 1) * args.batch_size, args.num_path)

        samples = torch.FloatTensor(train_data[start:end]).to(device) # n * K * p
        optimizer.zero_grad()
        output = net(samples)
        #print('output by FullNet',output)

        score = torch.mean(output)

        score.backward()
        optimizer.step()

        train_score += score.item() * len(samples)


    history['train_score'].append(train_score/args.num_path)
    training_time1 += time.time() - start_time
    print(train_score/args.num_path)


def test(epoch, net):
    global best_score
    net.eval()
    test_score = 0

    with torch.no_grad():
        num_batch_test = int(np.ceil(num_path_test / batch_size_test))

        for batch_idx in range(num_batch_test):
            start = batch_idx * batch_size_test
            end = np.minimum((batch_idx + 1) * batch_size_test, num_path_test)

            samples = torch.FloatTensor(test_data[start:end]).to(device)

            output = net(samples)
            score = torch.mean(output)

            test_score += score.item() * len(samples)

        history['test_score'].append(test_score/num_path_test)
        print(test_score/num_path_test)

    if test_score/num_path_test < best_score:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'score': test_score/num_path_test,
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        best_score = np.mean(test_score)

        if not os.path.isdir(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        torch.save(state, os.path.join(args.ckpt_dir, save))
    return np.mean(test_score)


num_path_test = 100000 # 1,000,000
batch_size_test = 5000
test_data = generate_path(args.asset_model, args.num_asset, args.m, num_path_test, time_step, R_f)

for epoch in range(1, args.epochs+1):
    train(epoch, net)
    test_score = test(epoch, net)

    if args.scheduler:
        if args.scheduler_type == 'auto':
            scheduler.step(np.mean(test_score))
        elif args.scheduler_type == 'step':
            adjust_learning_rate(optimizer, epoch, args.steplr, args.lr_gamma)

    if (args.beta_annealing) & (epoch % args.beta_step == 0):
        for param_group in optimizer.param_groups:
            param_group['beta'] *= args.beta_gamma







##save results
print(save)
print('best score...', best_score)
print('running time:', time.time()-start)
print('training time:', training_time1)

history['best_score'] = best_score
history['running_time'] = time.time()-start
history['training_time'] = training_time1


state = torch.load(open(os.path.join(args.ckpt_dir, save), 'rb'))
print('best_epoch', state['epoch'])

history['best_epoch'] = state['epoch']


if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
pkl.dump(history, open(os.path.join(args.log_dir, save), 'wb'))

