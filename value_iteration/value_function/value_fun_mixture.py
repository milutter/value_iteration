import torch
from torch.optim import Adam, SGD


class ValueFunctionMixture:
    def __init__(self, n_state, **kwargs):
        self.net = kwargs['val_class'](n_state, **kwargs)
        self.name = self.net.name + f"_mixture{self.net.n_network:02d}"
        self.device = self.net.device

    def __call__(self, x, fit=False):
        return self.forward(x, fit=fit)

    def forward(self, x, fit=False):
        V, dVdx = self.net(x)
        V, dVdx = (V, dVdx) if fit else (V.mean(dim=0), dVdx.mean(dim=0))
        return V, dVdx

    def cuda(self, device=None):
        self.net.cuda()
        self.device = self.net.device
        return self

    def cpu(self):
        self.net.cpu()
        self.device = self.net.device
        return self

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):

        try:
            self.net.load_state_dict(state_dict)

        except AttributeError:
            assert len(state_dict) == self.net.n_network

            new_state_dict = {}
            for key in state_dict[0].keys():
                stacked_parameters = torch.stack([dict_i[key] for dict_i in state_dict], dim=0)
                new_state_dict[key] = stacked_parameters

            self.net.load_state_dict(new_state_dict)


class AdamOptimizerMixture:
    def __init__(self, value_fun, lr=1.e-3, weight_decay=0.0, amsgrad=True):
        self.opt = [Adam(net_i.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad) for net_i in value_fun.net]

    def zero_grad(self):
        _ = [opt_i.zero_grad() for opt_i in self.opt]

    def step(self):
        _ = [opt_i.step() for opt_i in self.opt]