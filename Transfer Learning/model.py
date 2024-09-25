# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(777)

## package load
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import single_nn, two_nn, two_nn_tf
from custom_optim import TUSLA, SGLD, SGHMC
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--eta', default=1e-6, type=float) # regularization
parser.add_argument('--lr', default=0.01, type=float) # learning rate
parser.add_argument('--beta', default=1e10, type=float) # inverse temp
parser.add_argument('--gamma', default=0.5, type=float) # gamma
parser.add_argument('--epochs', default=100, type=int)
args = parser.parse_args()
torch.manual_seed(777)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

## generate dataset
n = 10000
n_train = int(n*0.7)

y = torch.rand(n, 2) #input variable
z = (np.abs(2 * y[:, 0] + 2 * y[:, 1] - 1.5)).pow(3)

y = y.to(device)
z = z.to(device)

class CustomDataset(Dataset):
  def __init__(self, y, z):
    self.y = y
    self.z = z
  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return y[idx], z[idx]

## HYPERPARAMETER SETTING
print('==================learning TransferLearning_FeedbackNetwork==================')
lr = args.lr
eta = args.eta # Regularization Para
beta = args.beta # Inverse temperature
gamma = args.gamma
epochs = args.epochs

#r = 0.5
dim = 30 # hidden size 30

batch_size = 128
act_fn = 'sigmoid' # 'sigmoid'
ckpt_dir = './model_save/'

traindataset = CustomDataset(y[:n_train], z[:n_train])
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

testdataset = CustomDataset(y[n_train:], z[n_train:])
testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)

file_name = 'lr_' + str(lr) + '_eta_' + str(eta) + '_'
fig_dir = './figures/' + file_name  +'/'


hist_train = {}
hist_test = {}
networks = {}
#
settings = [[eta, 'SGHMC']]
#
criterion = nn.MSELoss()

num_batch = np.ceil(traindataset.__len__()/batch_size).astype('int')
plt.figure(1)

for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    net = two_nn(dim, act_fn).to(device)
    print(net.parameters())
    if opt_name == 'ADAM': opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif opt_name == 'AMSGrad': opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)
    elif opt_name == 'SGD': opt = optim.SGD(net.parameters(), lr=0.01)
    elif opt_name == 'RMSprop': opt = optim.RMSprop(net.parameters(), lr=0.01)
    elif opt_name == 'TUSLA': opt = TUSLA(net.parameters(), lr=lr, eta=eta, beta=beta, r=r) # 相对平滑
    elif opt_name == 'SGLD': opt = SGLD(net.parameters(), lr=lr, beta=beta) # 相对平滑
    elif opt_name == 'SGHMC': opt = SGHMC(net.parameters(), lr=lr, beta=beta, gamma = gamma) # 有更强的跳出局部点的能力
    exp_name = opt_name + '_eta=' + str(eta)

    hist_train[exp_name] = []
    hist_test[exp_name] = []

    for epoch in range(1, epochs+1):
        train_loss = []
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            targets = targets.view(-1, 1)
            opt.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
        hist_train[exp_name].append(np.mean(train_loss))

        net.eval()
        test_loss = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.view(-1, 1)
            opt.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()
            test_loss.append(loss.item())
        hist_test[exp_name].append(np.mean(test_loss))

        print('epoch: %d, training_loss: %.8f, test_loss: %.8f'%(epoch, np.mean(train_loss), np.mean(test_loss)))

    networks[exp_name] = net

    state = {
        'net': net.state_dict(),
    }

    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    torch.save(state, '%s/%s.pth' % (ckpt_dir, exp_name))

    plt.plot(range(1, len(hist_train[exp_name]) + 1), hist_train[exp_name])
    plt.xlabel('epochs')
    plt.ylabel('train_loss')


plt.plot(range(1, len(hist_train[exp_name])+1), np.zeros(len(hist_train[exp_name])), 'k')

file_name = 'lr_' + str(lr) + '_eta_' + str(eta) + '_'
fig_dir = './figures/'# + file_name  +'/'
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
plt.savefig(fig_dir + file_name + 'first_nn_training.png')


plt.figure(2)
plt.plot(range(1, len(hist_train[exp_name])+1), np.zeros(len(hist_train[exp_name])), 'k')
for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    exp_name = opt_name + '_eta=' + str(eta)
    plt.plot(range(1, len(hist_train[exp_name]) + 1), hist_test[exp_name])#, label=name_lr)
plt.xlabel('epochs')
plt.ylabel('validation_loss')

plt.savefig(fig_dir + file_name + 'fist_nn_test.png')

plt.figure(3)
ax = plt.axes(projection='3d')
net.eval()
grid = torch.FloatTensor([[x/50, y/50] for x in range(0, 51) for y in range(0, 51)])
true = np.power(np.abs(2*grid[:, 0] + 2 * grid[:, 1] - 1.5), 3) # (np.abs(2 * y[:, 0] + 2 * y[:, 1] - 1.5)).pow(3)
ax.scatter(grid[:, 0], grid[:, 1], true,  alpha=0.5, label='true')
for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    exp_name = opt_name + '_eta=' + str(eta)
    net = networks[exp_name]
    pred = net(grid.to(device)).cpu().data
    ax.scatter(grid[:, 0], grid[:, 1], pred, alpha=0.5, label='prediction') #, label=name_lr)

plt.legend()
#plt.savefig(fig_dir + file_name + 'fist_nn_plot.png')

################################## transfer learning ##################################################################
print('=========================start transfer learning=========================')

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#def sigmoid(x):
#    x_ravel = x.ravel()  # 将numpy数组展平
#    length = len(x_ravel)
#    y = []
#    for index in range(length):
#        if x_ravel[index] >= 0:
#            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
#        else:
#            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
#    return np.array(y).reshape(x.shape)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, output_size, numpy_params):
        self.beta = 1e10    # Friction term
        self.gamma = 0.5  # Scale of random noise
        self.params = {
            'weights_hidden1': numpy_params['l1.weight'].T, #np.random.randn(input_size, hidden_layer1_size) * 10,
            'biases_hidden1': numpy_params['l1.bias'], #np.random.randn(hidden_layer1_size),
            'weights_hidden2': numpy_params['l2.weight'].T, # np.random.randn(hidden_layer1_size, hidden_layer2_size) * 10,
            'biases_hidden2': numpy_params['l2.bias'], # np.random.randn(hidden_layer2_size),
            'weights_output': numpy_params['l3.weight'].T # np.random.randn(hidden_layer2_size, output_size) * 10
        }
        self.velocity = {key: np.zeros_like(value) for key, value in self.params.items()}

        #self.weights_hidden1 = np.random.randn(input_size, hidden_layer1_size) * 10
        #self.biases_hidden1 = np.random.randn(hidden_layer1_size)
        #self.weights_hidden2 = np.random.randn(hidden_layer1_size, hidden_layer2_size) * 10
        #self.biases_hidden2 = np.random.randn(hidden_layer2_size)
        #self.weights_output = np.random.randn(hidden_layer2_size, output_size) * 10

    def forward_pass(self, X):
        self.hidden1_input = np.dot(X, self.params['weights_hidden1']) + 1 * np.tanh(self.params['biases_hidden1'])
        self.hidden1_output = relu(self.hidden1_input)

        self.hidden2_input = np.dot(self.hidden1_output, self.params['weights_hidden2']) + 1 * np.tanh(self.params['biases_hidden2'])
        self.hidden2_output = sigmoid(self.hidden2_input)

        self.output_layer_input = np.dot(self.hidden2_output, self.params['weights_output'])
        self.output = self.output_layer_input

        return self.output

    def backpropagate(self, X, y_true):
        m = y_true.shape[0]
        d_loss_output = (self.output - y_true.reshape(self.output.shape[0],-1)) / m

        d_weights_output = np.dot(self.hidden2_output.T, d_loss_output)

        d_hidden2 = np.dot(d_loss_output, self.params['weights_output'].T) * sigmoid_derivative(self.hidden2_input)
        d_weights_hidden2 = np.dot(self.hidden1_output.T, d_hidden2)
        d_biases_hidden2 = np.sum(d_hidden2, axis=0)

        d_hidden1 = np.dot(d_hidden2, self.params['weights_hidden2'].T) * relu_derivative(self.hidden1_input)
        d_weights_hidden1 = np.dot(X.T, d_hidden1)
        d_biases_hidden1 = np.sum(d_hidden1, axis=0) * (1 - np.tanh(self.params['biases_hidden1'])**2)

        gradients = {
            'weights_output': d_weights_output,
            'weights_hidden2': d_weights_hidden2,
            'biases_hidden2': d_biases_hidden2,
            'weights_hidden1': d_weights_hidden1,
            'biases_hidden1': d_biases_hidden1
        }
        return gradients


    def update_para_SGHMC(self, gradients, learning_rate=0.01):
        for key in self.velocity:
            # Introduce random noise for the stochastic part of SGHMC
            # self.velocity['biases_hidden1'].reshape(-1, 1).shape
            if 'biases' in key:
                noise = np.random.randn(self.velocity[key].reshape(-1, 1).shape[0], 1) * np.sqrt(2 * learning_rate * self.gamma / self.beta)
                self.velocity[key] = (1 - learning_rate * self.gamma) * np.squeeze(self.velocity[key]) - learning_rate * gradients[key] + np.squeeze(noise)
                # Update parameters
                self.params[key] += learning_rate * self.velocity[key]
            elif key =='weights_hidden2':
                noise = np.random.randn(self.velocity[key].shape[0], self.velocity[key].shape[1]) * np.sqrt(2 * learning_rate*self.gamma  / self.beta)
            #    # Update velocity
                self.velocity[key] = (1 - learning_rate * self.gamma) * self.velocity[key] - learning_rate * gradients[key] + noise
                # Update parameters
                self.params[key] += learning_rate * self.velocity[key]
            else:
                self.params[key] += 0
    def update_parameters(self, gradients, learning_rate):
        for key in self.params:
            self.params[key] -= learning_rate * gradients[key]
        #self.params['weights_output'] -= learning_rate * gradients['weights_output']
        #self.params['weights_hidden2'] -= learning_rate * gradients['weights_hidden2']
        #self.params['biases_hidden2'] -= learning_rate * gradients['biases_hidden2']
        #self.params['weights_hidden1'] -= learning_rate * gradients['weights_hidden1']
        #self.params['biases_hidden1'] -= learning_rate * gradients['biases_hidden1']

    def train(self, X_train, y_train, learning_rate=0.01, epochs=200, batch_size=128):
        num_samples = len(X_train)
        num_batches = num_samples // batch_size

        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                X_batch = X_train_shuffled[batch_start:batch_end]
                y_batch = y_train_shuffled[batch_start:batch_end]
                self.forward_pass(X_batch)
                gradients = self.backpropagate(X_batch, y_batch)
                if epoch > epochs / 2:
                    self.update_para_SGHMC(gradients, learning_rate*0.1)
                else:
                    self.update_para_SGHMC(gradients, learning_rate)


            if epoch % 1 == 0:
                y_pred = self.forward_pass(X_train)
                loss = mse_loss(y_train, y_pred)
                print(f'Epoch {epoch}, Loss: {loss}')


# Example usage:
# Define the neural network structure
input_size = 2  # Number of input features
hidden_layer1_size = 30
hidden_layer2_size = 30
output_size = 1  # Number of output features
n = 20000
# Create the neural network
state = torch.load('./model_save/SGHMC_eta=1e-06.pth' ) # ['net']
numpy_params = {k: np.array(v) for k, v in state['net'].items()}
nn_sghmc = NeuralNetwork(input_size, hidden_layer1_size, hidden_layer2_size, output_size, numpy_params)

# Generate some dummy data for training
X_train = np.random.rand(n, input_size)
y_train = -(1.5 * X_train[:, 0] + 0.5 * X_train[:, 1] - 1)**3

# Train the neural network using mini-batch SGHMC
nn_sghmc.train(X_train, y_train, learning_rate=0.05, epochs=100, batch_size=128)

# Predict on new data

plt.figure(6)
ax = plt.axes(projection='3d')
grid = np.array([[x/50, y/50] for x in range(0, 51) for y in range(0, 51)])
true = -(1.5 * grid[:, 0] + 0.5* grid[:, 1] - 1)**3
ax.scatter(grid[:, 0], grid[:, 1], true,  alpha=0.5, label='true')

pred = nn_sghmc.forward_pass(grid)
ax.scatter(grid[:, 0], grid[:, 1], pred, alpha=0.5, label='prediction')
plt.legend()
plt.show()