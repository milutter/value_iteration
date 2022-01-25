import numpy as np
import torch

def sigmoid(x, x0, a=1.0):
    z = torch.abs(x) - x0
    exp_z = torch.exp(-a * z)
    return 1. / (1. + exp_z)


class BarrierCost:
    def __init__(self, cost, x_penalty, cuda):
        self.n_state = x_penalty.shape[0]
        self.x_bnd_lim = x_penalty.clone().view(1, self.n_state, 1)
        self.x_bnd_lim = self.x_bnd_lim.cuda() if cuda else self.x_bnd_lim.cpu()

        self.q = cost
        self.q_lim = cost(torch.diag(self.x_bnd_lim.view(self.n_state)).view(self.n_state, self.n_state, 1)).view(1, self.n_state, 1)

    def __call__(self, x):
        numpy = isinstance(x, np.ndarray)

        x = torch.from_numpy(x).float().view(-1, self.n_state, 1) if numpy else x.view(-1, self.n_state, 1)
        out = self.q(x) + torch.sum(5. * self.q_lim * sigmoid(x, self.x_bnd_lim, a=20.), dim=1).view(x.shape[0], 1, 1)
        return out.numpy() if numpy else out

    def grad(self, x):
        f = sigmoid(x, self.x_bnd_lim, a=20.)
        dfdx = f * (1 - f)
        return self.q.grad(x) + torch.sum(5. * self.q_lim * dfdx, dim=1).view(x.shape[0], 1, 1)

    def cuda(self, device=None):
        self.q_lim = self.q_lim.cuda(device=device)
        self.x_bnd_lim = self.x_bnd_lim.cuda(device=device)
        self.q.cuda(device=device)
        return self

    def cpu(self):
        self.q_lim = self.q_lim.cpu()
        self.x_bnd_lim = self.x_bnd_lim.cpu()
        self.q.cpu()
        return self
