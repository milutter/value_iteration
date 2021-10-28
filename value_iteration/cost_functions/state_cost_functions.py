import torch
import numpy as np


class QuadraticCost:
    def __init__(self, Q, cuda=False, feature=None):
        self.n = Q.shape[-1]
        self.np_Q = Q.reshape((1, self.n, self.n))
        self.to_Q = torch.from_numpy(Q).view(1, self.n, self.n).float()

        self.device = None
        self.cuda() if cuda else self.cpu()

    def __call__(self, x):
        return self.numpy(x) if isinstance(x, np.ndarray) else self.torch(x)

    def grad(self, x):
        return self.grad_numpy(x) if isinstance(x, np.ndarray) else self.grad_torch(x)

    def hessian(self, x):
        return self.hessian_numpy(x) if isinstance(x, np.ndarray) else self.hessian_torch(x)

    def numpy(self, x):
        x = x.reshape((-1, self.n, 1))
        return np.matmul(x.transpose((0, 2, 1)), np.matmul(self.np_Q, x)).reshape((-1, 1, 1))

    def torch(self, x):
        x = x.view(-1, self.n, 1)
        return torch.matmul(x.transpose(1, 2), torch.matmul(self.to_Q, x)).view(-1, 1, 1)

    def grad_numpy(self, x):
        x = x.reshape((-1, self.n, 1))
        return 2. * np.matmul(self.np_Q, x).reshape((-1, self.n, 1))

    def grad_torch(self, x):
        x = x.view(-1, self.n, 1)
        return 2. * torch.matmul(self.to_Q, x).view(-1, self.n, 1)

    def hessian_numpy(self, x):
        return 2. * self.np_Q.repeat(x.shape[0], 1, 1)

    def hessian_torch(self, x):
        return 2. * self.to_Q.repeat(x.shape[0], 1, 1)

    def cuda(self, device=None):
        self.to_Q = self.to_Q.cuda()
        self.device = self.to_Q.device
        return self

    def cpu(self):
        self.to_Q = self.to_Q.cpu()
        self.device = self.to_Q.device
        return self


class FeatureQuadraticCost:
    def __init__(self, Q, feature=None, cuda=False):
        feature = feature if feature is not None else np.zeros(Q.shape[-1])
        feature = np.clip(feature, 0., 1.0)
        assert feature.size == Q.shape[-1]

        self.n = Q.shape[-1]
        self.np_Q = Q.reshape((1, self.n, self.n))
        self.to_Q = torch.from_numpy(Q).float().view(1, self.n, self.n)
        self.np_f = feature.reshape((1, self.n, 1))
        self.np_notf = (1. - feature).reshape((1, self.n, 1))
        self.to_f = torch.from_numpy(self.np_f).float().view(1, self.n, 1)
        self.to_notf = torch.from_numpy(self.np_notf).float().view(1, self.n, 1)

        self.device = None
        self.cuda() if cuda else self.cpu()

    def __call__(self, x):
        return self.numpy(x) if isinstance(x, np.ndarray) else self.torch(x)

    def grad(self, x):
        return self.grad_numpy(x) if isinstance(x, np.ndarray) else self.grad_torch(x)

    def hessian(self, x):
        return self.hessian_numpy(x) if isinstance(x, np.ndarray) else self.hessian_torch(x)

    def feature_numpy(self, x):
        return x

    def feature_torch(self, x):
        return x

    def feature_casadi(self, x):
        return x

    def grad_feature_numpy(self, x):
        return np.ones_like(x)

    def grad_feature_torch(self, x):
        return torch.ones_like(x).to(x.device())

    def hesssian_feature_numpy(self, x):
        return np.zeros_like(x)

    def hessian_feature_torch(self, x):
        return torch.zeros_like(x).to(x.device())

    def numpy(self, x):
        x = x.reshape((-1, self.n, 1))
        f = self.np_notf * x + self.np_f * self.feature_numpy(x)
        return np.matmul(f.transpose((0, 2, 1)), np.matmul(self.np_Q, f)).reshape((-1, 1, 1))

    def torch(self, x):
        x = x.reshape((-1, self.n, 1))
        f = self.to_notf * x + self.to_f * self.feature_torch(x)
        return torch.matmul(f.transpose(1, 2), torch.matmul(self.to_Q, f)).view(-1, 1, 1)

    def grad_numpy(self, x):
        # Let q = f^T Q f with f = f(x)
        # =>  dq/dx = 2 Q f (*) df/dx where (*) refers to elementwise multiplication
        x = x.reshape((-1, self.n, 1))
        f = self.np_notf * x + self.np_f * self.feature_numpy(x)
        dfdx = (self.np_notf *  np.ones_like(x) + self.np_f * self.grad_feature_numpy(x)).reshape((-1, self.n, 1))
        return 2. * np.matmul(self.np_Q, f).reshape((-1, self.n, 1)) * dfdx

    def grad_torch(self, x):
        x = x.reshape((-1, self.n, 1))
        f = self.to_notf * x + self.to_f * self.feature_torch(x)
        dfdx = (self.to_notf * torch.ones_like(x) + self.to_f * self.grad_feature_torch(x)).view(-1, self.n, 1)
        return 2. * torch.matmul(self.to_Q, f).view(-1, self.n, 1) * dfdx

    def hessian_torch(self, x):
        x = x.reshape((-1, self.n, 1))
        f = self.to_notf * x + self.to_f * self.feature_torch(x)
        dfdx = (self.to_notf * torch.ones_like(x) + self.to_f * self.grad_feature_torch(x)).view(-1, self.n, 1)
        d2fd2x = torch.diag_embed((self.to_notf * torch.zeros_like(x) + self.to_f * self.hessian_feature_torch(x)).squeeze(-1))
        return 2. * (self.to_Q * torch.matmul(dfdx, dfdx.transpose(dim0=1, dim1=2)) +torch.matmul(self.to_Q, f).view(-1, self.n, 1) * d2fd2x)

    def cuda(self, device=None):
        self.to_Q = self.to_Q.cuda()
        self.to_f = self.to_f.cuda()
        self.to_notf = self.to_notf.cuda()
        self.device = self.to_Q.device
        return self

    def cpu(self):
        self.to_Q = self.to_Q.cpu()
        self.to_f = self.to_f.cpu()
        self.to_notf = self.to_notf.cpu()
        self.device = self.to_Q.device
        return self


class SineQuadraticCost(FeatureQuadraticCost):
    def __init__(self, Q, feature=None, cuda=False):
        super(SineQuadraticCost, self).__init__(Q, feature, cuda)

    def feature_numpy(self, x):
        return np.pi * np.sin(x/2.)

    def feature_torch(self, x):
        return np.pi * torch.sin(x/2.)

    def grad_feature_numpy(self, x):
        return np.pi/2. * np.cos(x/2.)

    def grad_feature_torch(self, x):
        return np.pi/2. * torch.cos(x/2.)

    def hessian_feature_numpy(self, x):
        return -np.pi/4. * np.sin(x/2.)

    def hessian_feature_torch(self, x):
        return -np.pi / 4. * torch.sin(x / 2.)


class CosineQuadraticCost(FeatureQuadraticCost):
    def __init__(self, Q, feature=None, cuda=False):
        super(CosineQuadraticCost, self).__init__(Q, feature, cuda)

    def feature_numpy(self, x):
        return np.pi * (1. - np.cos(x / 2.))

    def feature_torch(self, x):
        return np.pi * (1. - torch.cos(x / 2.))

    def feature_casadi(self, x):
        return ca.pi * (1. - ca.cos(x / 2.))

    def grad_feature_numpy(self, x):
        return np.pi / 2. * np.sin(x / 2.)

    def grad_feature_torch(self, x):
        return np.pi / 2. * torch.sin(x / 2.)


if __name__ == "__main__":
    n = 100
    Q = np.diag(np.array([1.0, 0.1, 0.5, 0.2]))

    x0 = np.hstack([np.linspace(-np.pi, np.pi, n, endpoint=True)[:, np.newaxis], np.zeros(n)[:, np.newaxis]])

    x0 = np.hstack([np.linspace(-np.pi, np.pi, n, endpoint=True)[:, np.newaxis],
                    np.linspace(-5., +5., n, endpoint=True)[:, np.newaxis],
                    np.linspace(-5., +5., n, endpoint=True)[:, np.newaxis],
                    np.linspace(-5., +5., n, endpoint=True)[:, np.newaxis]])

    feature = np.array([1., 0., 1., 0.])

    fc_quad = QuadraticCost(Q)
    fc_feat = FeatureQuadraticCost(Q, feature)
    fc_sine = SineQuadraticCost(Q, feature)
    fc_cosi = CosineQuadraticCost(Q, feature)

    c_quad, dcdx_quad = fc_quad(x0), fc_quad.grad(x0)
    c_feat, dcdx_feat = fc_feat(x0), fc_feat.grad(x0)
    c_sine, dcdx_sine = fc_sine(x0), fc_sine.grad(x0)
    c_cosi, dcdx_cosi = fc_cosi(x0), fc_cosi.grad(x0)

    torch_x0 = torch.from_numpy(x0)
    torch_c_quad, torch_dcdx_quad = fc_quad(torch_x0).detach().numpy(), fc_quad.grad(torch_x0).detach().numpy()
    torch_c_feat, torch_dcdx_feat = fc_feat(torch_x0).detach().numpy(), fc_feat.grad(torch_x0).detach().numpy()
    torch_c_sine, torch_dcdx_sine = fc_sine(torch_x0).detach().numpy(), fc_sine.grad(torch_x0).detach().numpy()
    torch_c_cosi, torch_dcdx_cosi = fc_cosi(torch_x0).detach().numpy(), fc_cosi.grad(torch_x0).detach().numpy()

    assert np.allclose(torch_c_quad, c_quad) and np.allclose(torch_dcdx_quad, dcdx_quad)
    assert np.allclose(torch_c_feat, c_feat) and np.allclose(torch_dcdx_feat, dcdx_feat)
    assert np.allclose(torch_c_sine, c_sine) and np.allclose(torch_dcdx_sine, dcdx_sine)
    assert np.allclose(torch_c_cosi, c_cosi) and np.allclose(torch_dcdx_cosi, dcdx_cosi)

    import matplotlib as mp
    mp.use("Qt5Agg")
    from matplotlib import cm
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = [fig.add_subplot(1, feature.size+1, i+1) for i in range(feature.size+1)]
    ax[0].plot(x0[:, 0], c_quad[:, 0, 0], c=cm.Set1(0), label="Quadratic Cost")
    ax[0].plot(x0[:, 0], c_feat[:, 0, 0], c=cm.Set1(1), label="Feature Cost")
    ax[0].plot(x0[:, 0], c_sine[:, 0, 0], c=cm.Set1(2), label="Sine Cost")
    ax[0].plot(x0[:, 0], c_cosi[:, 0, 0], c=cm.Set1(3), label="Cosine Cost")
    ax[0].legend()

    for i in range(feature.size):
        ax[i+1].plot(x0[:, 0], dcdx_quad[:, i, 0], c=cm.Set1(0))
        ax[i+1].plot(x0[:, 0], dcdx_feat[:, i, 0], c=cm.Set1(1))
        ax[i+1].plot(x0[:, 0], dcdx_sine[:, i, 0], c=cm.Set1(2))
        ax[i+1].plot(x0[:, 0], dcdx_cosi[:, i, 0], c=cm.Set1(3))

    plt.show()