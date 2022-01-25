import torch
import argparse
import numpy as np

try:
    import matplotlib as mpl
    mpl.use("Qt5Agg")

except ImportError as e:
    pass


from value_iteration.value_function import QuadraticNetwork, TrigonometricQuadraticNetwork
from value_iteration.run_experiment import run_experiment
from value_iteration.cartpole import CartpoleLogCos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-alg", dest='algorithm', type=str, default="cFVI", required=False, help="Specify the Algorithm.")
    parser.add_argument("-seed", dest='seed', type=int, default=42, required=False, help="Specifies the random seed")
    parser.add_argument("-load", dest='load', type=int, default=1, required=False, help="Specifies whether to load an existing model.")
    args = parser.parse_args()
    assert args.algorithm.lower() in ["cfvi", "rfvi"]

    # Initialize NumPy:
    np.random.seed(args.seed)
    np.set_printoptions(
        suppress=True, precision=2, linewidth=500,
        formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # Initialize PyTorch:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    model_path = None
    if args.algorithm.lower() == 'rfvi' and bool(args.load):
        model_path = 'data/rFVI_Cartpole_LogCosCost.torch'

    if args.algorithm.lower() == 'cfvi' and bool(args.load):
        model_path = 'data/cFVI_Cartpole_LogCosCost.torch'


    # Define Hyper-parameters:
    hyper = {
        # Learning Mode:
        'mode': 'DP',
        'robust': True if args.algorithm.lower() == 'rfvi' else False,

        # Value Function:
        'val_class': TrigonometricQuadraticNetwork,
        'checkpoint': model_path,
        'plot': False,

        # System Specification:
        'system_class': CartpoleLogCos,
        'state_cost': '25.e+0, 1.e+0, 5.e-1, 1.e-1',
        'action_cost': '1.e-1',
        'eps': 0.80,  # eps = 1 => \gamma = 1
        'dt': 1. / 125.,
        'T': 5.,

        # Network:
        'n_network': 4,
        'activation': 'Tanh',
        'n_width': 64,
        'n_depth': 4,
        'n_output': 1,
        'g_hidden': 1.41,
        'g_output': 1.,
        'b_output': -0.1,

        # Samples
        'n_iter': 250,
        'eval_minibatch': 256 * 200,
        'test_minibatch': 256 * 20,
        'n_minibatch': 512,
        'n_batches': 200,

        # Network Optimization
        'max_epoch': 20,
        'lr_SGD': 1.0e-4,
        'weight_decay': 1.e-6,
        'exp': 1.,

        # Lambda Traces
        'trace_weight_n': 1.e-4,
        'trace_lambda': 0.90,

        # Exploration:
        'x_noise': 1.e-6,
        'u_noise': 1.e-6,
    }

    # Select the admissible set of the adversary:
    hyper['xi_x_alpha'] = 0.025 if hyper["robust"] else 1.e-6
    hyper['xi_u_alpha'] = 0.100 if hyper["robust"] else 1.e-6
    hyper['xi_o_alpha'] = 0.025 if hyper["robust"] else 1.e-6
    hyper['xi_m_alpha'] = 0.150 if hyper["robust"] else 1.e-6

    print("Hyperparameters:")
    for key in hyper.keys():
        print(f"{key:30}: {hyper[key]}")

    # Run Experiment:
    run_experiment(hyper)
