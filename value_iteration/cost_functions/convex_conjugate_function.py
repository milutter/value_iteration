import numpy as np
import torch as torch


class ConvexConjugateFunction:
    def __call__(self, *args, **kwargs):
        pass

    def grad(self, x):
        pass

    def convex_conjugate(self, x):
        pass

    def grad_convex_conjugate(self, x):
        pass


class QuadraticFunction(ConvexConjugateFunction):
    def __init__(self, A, cuda=False, domain=(-1., +1.)):

        self.domain = domain
        self.convex_domain = domain
        self.n = A.shape[1]
        self.A = A
        self.invA = np.linalg.inv(A)

        self.A_torch = torch.from_numpy(self.A).float().view(-1, self.n, self.n)
        self.invA_torch = torch.from_numpy(self.invA).float()

        self.device = None
        self.cuda() if cuda else self.cpu()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = x.reshape((-1, self.n, 1))
            quad = np.matmul(x.transpose(0, 2, 1), np.matmul(self.A.reshape((1, self.n, self.n)), x)).reshape(x.shape[0], 1, 1)

        elif isinstance(x, torch.Tensor):
            # shape = x.shape
            x = x.view(-1, self.n, 1)
            quad = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), self.A_torch), x).view(x.shape[0], 1, 1)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return quad

    def grad(self, x):
        if isinstance(x, np.ndarray):
            shape = x.shape
            x = x.reshape((-1, self.n, 1))
            Ax = np.matmul(self.A.reshape((1, self.n, self.n)), x).reshape(shape)

        elif isinstance(x, torch.Tensor):
            x = x.view(-1, self.n, 1)
            Ax = torch.matmul(x.transpose(dim0=1, dim1=2), self.A_torch).squeeze()

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return 2. * Ax

    def convex_conjugate(self, x):
        if isinstance(x, np.ndarray):
            shape = x.shape
            x = x.reshape((-1, self.n, 1))
            xTinvAx = np.matmul(x.transpose(0, 2, 1), np.matmul(self.invA.reshape((1, self.n, self.n)), x)).reshape(shape)

        elif isinstance(x, torch.Tensor):
            shape = x.shape
            x = x.view(-1, self.n, 1)
            xTinvAx = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), self.invA_torch), x).view(*shape)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return 1./4. * xTinvAx

    def grad_convex_conjugate(self, x):
        if isinstance(x, np.ndarray):
            x = x.reshape((-1, self.n, 1))
            invAx = np.matmul(self.invA.reshape((1, self.n, self.n)), x)

        elif isinstance(x, torch.Tensor):
            shape = x.shape
            x = x.view(-1, self.n, 1)
            invAx = torch.matmul(x.transpose(dim0=1, dim1=2), self.invA_torch).view(*shape)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return 1./2. * invAx

    def cuda(self, device=None):
        self.A_torch = self.A_torch.cuda()
        self.invA_torch = self.invA_torch.cuda()
        self.device = self.A_torch.device
        return self

    def cpu(self):
        self.A_torch = self.A_torch.cpu()
        self.invA_torch = self.invA_torch.cpu()
        self.device = self.A_torch.device
        return self


class HyperbolicTangent(ConvexConjugateFunction):
    def __init__(self, alpha=+1., beta=+1.0, cuda=False):
        self.n = 1
        self.a = alpha
        self.b = beta
        self.domain = (-np.abs(alpha), +np.abs(alpha))
        self.convex_domain = (-10.0, +10.0)
        assert self.a >= self.domain[0] and self.a <= self.domain[1]

        # Compute Offset:
        self.off = 0.0
        self.off = -self(np.zeros(1))

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            assert np.all((self.a - np.abs(x)) >= 0.0)

            g = np.ones(x.size) * self.b * self.a * np.log(2. * self.a) + self.off
            mask = self.a - np.abs(x) > 0
            g[mask] = 0.5 * self.b * ((self.a - x[mask]) * np.log(self.a - x[mask]) +
                                      (self.a + x[mask]) * np.log(self.a + x[mask])) + self.off

        elif isinstance(x, torch.Tensor):
            x = x.view(-1, self.n, 1)
            g = 0.5 * self.b * (np.matmul((self.a - x).transpose(dim0=1, dim1=2), np.log(self.a - x)) +
                                np.matmul((self.a + x).transpose(dim0=1, dim1=2), np.log(self.a + x))) + self.off

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g.squeeze()

    def grad(self, x):
        if isinstance(x, np.ndarray):
            g_grad = self.b * np.arctanh(x/self.a)

        elif isinstance(x, torch.Tensor):
            g_grad = self.b * 0.5 * torch.log((1+x/self.a)/(1-x/self.a))

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g_grad

    def convex_conjugate(self, x):
        if isinstance(x, np.ndarray):
            # Naive implementation:
            g_star = self.a * self.b * np.log(np.cosh(x / self.b))

            # Numerically stable implementation to prevent overflows of exp(|x|):
            g_star = self.a * self.b * (np.log(0.5) + np.abs(x / self.b) +
                                        np.log(np.exp(-2. * np.abs(x / self.b)) + 1.0))

        elif isinstance(x, torch.Tensor):
            # Naive implementation:
            # g_star = self.a * self.b * torch.log(torch.cosh(x / self.b))

            # Numerically stable implementation to prevent overflows of exp(|x|):
            g_star = self.a * self.b * (torch.log(torch.tensor(0.5)) + torch.abs(x/self.b) +
                                        torch.log(torch.exp(-2. * torch.abs(x / self.b)) + 1.0))

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g_star

    def grad_convex_conjugate(self, x):
        if isinstance(x, np.ndarray):
            g_star_grad = self.a * np.tanh(x / self.b)

        elif isinstance(x, torch.Tensor):
            g_star_grad = self.a * torch.tanh(x / self.b)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g_star_grad


class ArcTangent(ConvexConjugateFunction):
    def __init__(self, alpha=+1., beta=+1.0):
        self.n = 1
        self.a = alpha
        self.b = beta
        self.domain = (-np.abs(alpha), +np.abs(alpha))
        self.convex_domain = (-10.0, +10.0)
        assert self.a >= self.domain[0] and self.a <= self.domain[1]

        # Add for numerical stability
        self.a += 1.e-3

        # Compute Offset:
        self.off = 0.0
        self.off = -self(np.zeros(1))

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            assert np.all(np.abs(x) <= self.a)

            shape = x.shape
            x = x.reshape((-1, self.n, 1))
            g = -2.0 * self.b * self.a / np.pi * np.log(np.clip(np.cos(np.pi / (2. * self.a) * x), 0.0, 1.0)).reshape(shape)

        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -self.a, self.a)
            assert torch.all(torch.abs(x) <= self.a)

            shape = x.shape
            x = x.view(-1, self.n, 1)
            g = -2.0 * self.b * self.a / np.pi * torch.log(torch.clamp(torch.cos(np.pi / (2. * self.a) * x), 0.0, 1.0)).view(shape)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g

    def grad(self, x):
        if isinstance(x, np.ndarray):
            assert np.all(np.abs(x) <= self.a)

            shape = x.shape
            x = x.reshape((-1, self.n, 1))
            g_grad = self.b * np.tan(np.pi/(2.*self.a) * x).reshape(shape)

        elif isinstance(x, torch.Tensor):
            assert torch.all(torch.abs(x) <= self.a)

            shape = x.shape
            x = x.view(-1, self.n, 1)
            g_grad = self.b * torch.tan(np.pi/(2.*self.a) * x).view(shape)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g_grad

    def convex_conjugate(self, x):
        if isinstance(x, np.ndarray):
            shape = x.shape
            x = x.reshape((-1, self.n, 1))
            g_star = self.a / np.pi * (2. * x * np.arctan(x/self.b)
                                       - self.b * np.log(self.b**2. + x**2.)
                                       + self.b * np.log(self.b**2)).reshape(shape)

        elif isinstance(x, torch.Tensor):
            shape = x.shape
            x = x.view(-1, self.n, 1)
            g_star = self.a / np.pi * (2. * x * torch.atan(x/self.b)
                                       - self.b * torch.log(self.b**2. + x**2.)
                                       + self.b * torch.log(torch.tensor(self.b**2))).view(shape)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g_star

    def grad_convex_conjugate(self, x):
        if isinstance(x, np.ndarray):

            shape = x.shape
            x = x.reshape((-1, self.n, 1))
            g_star_grad = 2. * self.a / np.pi * np.arctan(x / self.b).reshape(shape)

        elif isinstance(x, torch.Tensor):

            shape = x.shape
            x = x.view(-1, self.n, 1)
            g_star_grad = 2. * self.a / np.pi * torch.atan(x / self.b).view(shape)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g_star_grad

    def hessian_convex_conjugate(self, x):
        if isinstance(x, np.ndarray):

            shape = x.shape
            x = x.reshape((-1, self.n, 1))
            g_star_hessian = (2. * self.a * self.b) / (np.pi * (self.b**2 + x**2)).reshape(shape)

        elif isinstance(x, torch.Tensor):

            shape = x.shape
            x = x.view(-1, self.n, 1)
            g_star_hessian = (2. * self.a * self.b) / (np.pi * (self.b**2 + x**2)).view(shape)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))

        return g_star_hessian


def convex_conjugate_test(fun, verbose=True, plot=True):
    n, eps = 5000000, 1.e-3
    x, dx = np.linspace(fun.domain[0] + eps, fun.domain[1] - eps, n, endpoint=True, retstep=True)
    y, dy = np.linspace(fun.convex_domain[0] + eps, fun.convex_domain[1] - eps, n, endpoint=True, retstep=True)

    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()

    g = fun(x)
    g_star = fun.convex_conjugate(y)
    g_torch = fun(x_torch).numpy()
    g_star_torch = fun.convex_conjugate(y_torch).numpy()
    g_x0 = fun(0.0 * x[0:1])
    g_star_x0 = fun(0.0 * x[0:1])
    assert np.isclose(g_x0, 0.0) and np.isclose(g_star_x0, 0.0)

    g_grad = fun.grad(x)
    g_grad_torch = fun.grad(x_torch).numpy()
    g_grad_finite = np.diff(g) / np.diff(x)
    g_grad_err = np.mean(np.abs(g_grad[:-1] - g_grad_finite))

    g_star_grad = fun.grad_convex_conjugate(y)
    g_star_grad_torch = fun.grad_convex_conjugate(y_torch).numpy()
    g_star_grad_finite = np.diff(g_star) / np.diff(y)
    g_star_grad_err = np.mean(np.abs(g_star_grad[:-1] - g_star_grad_finite))

    err_g_torch = np.mean(np.abs(g - g_torch))
    err_g_star_torch = np.mean(np.abs(g_star - g_star_torch))
    err_g_grad_torch = np.mean(np.abs(g_grad - g_grad_torch))
    err_g_star_grad_torch = np.mean(np.abs(g_star_grad_torch - g_star_grad))

    inv_err = np.mean(np.abs(x - fun.grad_convex_conjugate(g_grad)))

    if verbose:
        print("\n\n{0}: ".format(str(fun.__class__).split(".")[1][:-2]))
        print("\t|diff g  - grad g |      = {0:.3e}".format(g_grad_err))
        print("\t|diff g* - grad g*|      = {0:.3e}".format(g_star_grad_err))
        print("\t|x - grad g*(grad g(x))| = ", end="")
        print("{0:.3e}".format(inv_err))

    if plot:
        import matplotlib as mp
        mp.use("Qt5Agg")
        import matplotlib.pyplot as plt
        n_down = 10000

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(x, g, color="r", label="g(x)")
        ax.plot(y, g_star, color="b", label="g*(x)")

        ax = fig.add_subplot(122)
        ax.plot(x[::n_down], g_grad[::n_down], color="r", label="g(x)", marker='s')
        ax.plot(x[:-1:n_down], g_grad_finite[::n_down], color="r", ls='--', marker='x')

        ax.plot(y[::n_down], g_star_grad[::n_down], color="b", label="g*(x)", marker='s')
        ax.plot(y[:-1:n_down], g_star_grad_finite[::n_down], color="b", ls='--', marker='x')

        plt.show()

    return g_grad_err < 5.e-3 and \
           g_star_grad_err < 1.e-3 and \
           inv_err < 1.e-3 and \
           err_g_torch < 1.e-3 and \
           err_g_star_torch < 1.e-3 and \
           err_g_grad_torch < 1.e-3 and \
           err_g_star_grad_torch < 1.e-3


if __name__ == "__main__":

    alpha = 5.0
    beta = -5.
    f = ArcTangent(alpha, beta)
    assert convex_conjugate_test(f, verbose=True, plot=False)

    alpha = 5.0
    beta = -2.
    f = HyperbolicTangent(alpha, beta)
    assert convex_conjugate_test(f, verbose=True, plot=False)

    A = np.diag(np.array([-1.7]))
    f = QuadraticFunction(A)
    assert convex_conjugate_test(f, verbose=True, plot=False)