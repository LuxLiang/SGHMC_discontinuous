import torch
import math
from torch.optim.optimizer import Optimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SGHMC(Optimizer):
    # Learning rate can't be too high

    def __init__(self, params, lr=1e-2, beta=1e10, gamma=0.5, weight_decay=0):

        defaults = dict(lr=lr, beta=beta, gamma=gamma, weight_decay=weight_decay)
        super(SGHMC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGHMC, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):

            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    grad = p.grad

                    if group['weight_decay'] != 0:
                        grad.add_(group['weight_decay'], p.data)

                    if len(state) == 0:
                        state['step'] = 0
                        state["vol"] = torch.zeros_like(p) # initial value = 0

                    beta, lr, gamma = group['beta'], group['lr'], group['gamma']
                    if group['weight_decay'] != 0:
                        grad.add_(group['weight_decay'], p.data)

                    state["step"] += 1

                    noise = math.sqrt(2 * lr * gamma / beta) * torch.randn(size=p.size(), device=device)
                    state["vol"] = state["vol"].add_(-lr * gamma * state["vol"]).add_(-lr * grad).add_(noise)
                    p.data.add_(lr * state["vol"])

            return loss