import sys
import time
import torch
import numpy as np
from tqdm import trange as tqdm_trange


def evaluate(fun, x, **kwargs):
    try:
        out = fun(x, **kwargs)

    except RuntimeError:
        idx = torch.linspace(0, x.shape[0], 3, dtype=torch.int)
        fx = zip(*[evaluate(fun, x[idx[i]:idx[i + 1]]) for i in range(idx.shape[0] - 1)])
        out = [torch.cat(x, dim=0) for x in fx]

    return out


def analyze_error(x):
    return torch.min(x), torch.median(x), torch.mean(x), torch.std(x), torch.max(x)


def error_statistics_string(x):
    if isinstance(x, tuple):
        out = f"{x[0].item():.1e} / {x[1].item():.1e} / {x[2].item():.1e} \u00B1 {1.96 * x[3].item():.1e} / {x[4].item():.1e}"

    elif isinstance(x, torch.Tensor):
        out = error_statistics_string(analyze_error(x))

    else:
        raise ValueError

    return out


def add_nan(x, wrap_i=0):
    th = 1.5

    if isinstance(x, list):
        new_list = []
        for j in range(len(x)):
            last_idx = 0
            x_new = np.zeros((0, x[j].shape[1]))

            for i in range(x[j].shape[0] - 1):
                if np.abs(x[j][i + 1, wrap_i] - x[j][i, wrap_i]) > th * np.pi:
                    x_new = np.vstack((x_new, x[j][last_idx:i + 1], np.nan * np.ones(x[j].shape[1]).reshape((1, x[j].shape[1]))))
                    last_idx = i + 1

            x_new = np.vstack((x_new, x[j][last_idx:]))
            new_list.append(x_new)

        out = new_list

    elif isinstance(x, np.ndarray):
        x_new = np.zeros((0, x.shape[1]))
        last_idx = 0

        for i in range(x.shape[0] - 1):
            if np.abs(x[i + 1, wrap_i] - x[i, wrap_i]) > th * np.pi:
                x_new = np.vstack((x_new, x[last_idx:i + 1], np.nan * np.ones((1, x.shape[1]))))
                last_idx = i + 1

        x_new = np.vstack((x_new, x[last_idx:]))
        out = x_new

    else:
        raise ValueError

    return out


def linspace(start, stop, n):
    if n == 1:
        out = torch.linspace((stop + start)/2., stop, n)

    else:
        out = torch.linspace(start, stop, n)

    return out


def negative_definite(*arg):
    if len(arg) == 2:
        f, dfdx = arg
        f_neg = torch.clamp(f, max=0.0)
        out = f_neg, dfdx

    else:
        f, dfdx, d2fd2x = arg
        f_neg = torch.clamp(f, max=0.0)
        out = f_neg, dfdx, d2fd2x

    return out


class trange:
    def __init__(self, n, verbose=False, prefix="", unit="it", ncols=125):
        self.verbose = verbose
        self.t0 = time.perf_counter()
        self.iter = tqdm_trange(n, desc=prefix, unit=unit, ncols=ncols, file=sys.stdout) if self.verbose else range(n)

    def __iter__(self):
        self.iterator = self.iter.__iter__()
        return self

    def __next__(self):
        if self.verbose:
            self.iter.set_postfix({"T": f"{time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - self.t0))}"})

        return self.iterator.__next__()
