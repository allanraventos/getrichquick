import multiprocessing
import os
import time
from functools import partial

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    StudentNetwork, TeacherNetwork, compute_hamming_distance, fit_nn_gd, get_alpha, get_kernel_trajectory, kernel_distance_from_initial, neuron, spherical_coordinates, test_one_hidden_layer_relu)


def get_results(scale, delta, seed, lr, n_iter, n_samples, m, input_size, leak_parameter, checkpoints_to_save):
    assert leak_parameter is None

    results = {}

    # This should work ok in a subprocess
    torch.manual_seed(seed)
    m0 = 3
    teacher = TeacherNetwork(input_size, m0)

    # Instantiate student model (TODO: change symmetrize)
    alpha = get_alpha(delta, scale)
    student = StudentNetwork(input_size, m, scale, alpha=alpha, symmetrize=True, leak_parameter=leak_parameter)

    # Sample data
    inputs = torch.randn(n_samples, input_size)
    inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
    with torch.no_grad():
        labels = teacher(inputs)
    # print(f"input mean: {inputs.mean()}, label mean: {labels.mean()}")

    # Extract initial student network paramters
    with torch.no_grad():
        W0, a0 = student.fc1.weight.detach().clone().numpy(), student.fc2.weight.detach().clone().numpy()
    
    W0, a0 = np.float64(W0), np.float64(a0)

    # Try training NN with numpy
    t0 = time.time()
    _results, losses_relu, preds_relu = fit_nn_gd(
        W0, a0, np.float64(inputs.detach().clone().numpy()), np.float64(labels.detach().clone().numpy()), leak_parameter=leak_parameter, lr=lr, n_iter=n_iter, checkpoints_to_save=checkpoints_to_save)
    t1 = time.time()
    print(f"Training took {t1 - t0} s")
    Ws_relu, as_relu = neuron(_results, input_size, m)

    K = get_kernel_trajectory(Ws_relu, as_relu, np.float64(inputs.detach().clone().numpy()), mode=None)
    kernel_distance = kernel_distance_from_initial(K)  # first one should be ~0

    test_losses = np.zeros(len(checkpoints_to_save))
    for _ci in range(len(checkpoints_to_save)):
        test_losses[_ci] = test_one_hidden_layer_relu(Ws_relu[_ci], as_relu[_ci], teacher, n_test=10_000, test_seed=201, leak_parameter=None)

    # Saving parameters (at all times not just checkpoints), train losses, train data, kernel trajectory, kernel distance, and test losses.
    # NOTE: with these we can compute/plot basically anything.
    # results["nn"] = (Ws_relu, as_relu, losses_relu, np.float64(inputs.detach().clone().numpy()), K, kernel_distance, test_losses)

    hamming_distance = compute_hamming_distance(Ws_relu, np.float64(inputs.detach().clone().numpy()))  # first one should be ~0
    parameter_distance = np.sqrt(((Ws_relu - Ws_relu[0][None])**2).sum(axis=(1, 2)) + ((as_relu - as_relu[0][None])**2).sum(axis=1))

    # May 31, 2024:
    # results["nn"] = (kernel_distance, test_losses)

    # New. NOTE: losses are every 1_000 by default; manually changing to every 100
    results["nn"] = (kernel_distance, test_losses, hamming_distance, parameter_distance, losses_relu)

    print(f"Rest (including test) took {time.time() - t1} s")

    t0 = time.time()
    # NOTE: here "slower" meant lower learning rate of 5e-2
    torch.save(results, f"dummy_results_longer/{scale}_{delta}_{seed}.pth")
    print(f"Saving took {time.time() - t0} s")

    return results


if __name__ == "__main__":
    n_cpus = multiprocessing.cpu_count()
    print(f"We have access to {n_cpus} cores")

    scales = np.logspace(-1, 0.3, 17)  # 11 points before
    deltas = np.linspace(-1, 1, 17)

    train_seeds = np.arange(100, 116)

    m = 50
    n = 1_000
    d = 100
    n_iter = 1_000_000  # May 31: running for 1M instead of the previous 100k
    base_lr = 5e-3  # slowed down to 5e-3 from 5e-2

    # Will save both kernel distances and test losses
    save_filename = f"m={m}_n={n}_d={d}_n_iter={n_iter}_base_lr={base_lr}_seeds_{train_seeds[0]}_{train_seeds[-1]}"

    # Construct arguments to run sweep with. NOTE that each seed is used for the whole grid
    args = []
    for scale in scales:
        for delta in deltas:
            for seed in train_seeds:
                args.append((scale, delta, seed, base_lr / scale**2))

    n_machines_run = min(len(args), n_cpus)
    print(f"Total number of jobs is: {len(args)}. Running on {n_machines_run} cores")

    # Could also change code to save these as well as every thousand, eh.
    checkpoints_to_save = (0, 10, 100, 1_000, 10_000, 100_000, 1_000_000)  # NOTE: added 1M

    with multiprocessing.Pool(n_machines_run) as _p:
        out = _p.starmap(
            partial(get_results, n_iter=n_iter, n_samples=n, m=m, input_size=d, leak_parameter=None, checkpoints_to_save=checkpoints_to_save), args)

    print(f"Done with multiprocessing.")

    # with multiprocessing.Pool(n_machines_run) as _p:
    #     with tqdm(total=len(args)) as pbar:
    #         out = []
    #         for result in _p.starmap(partial(get_results, n_iter=n_iter, n_samples=n, m=m, input_size=d, leak_parameter=None, checkpoints_to_save=checkpoints_to_save), args):
    #             pbar.update(1)
    #             out.append(result)

    # Collect results into a dictionary
    results = {}
    for _arg, _results in zip(args, out):
        # Just leave the learning rate out of the dictionary
        results[tuple(_arg[:-1])] = _results

    # May 31; commented out
    # torch.save(results, f"{save_filename}.pth")
