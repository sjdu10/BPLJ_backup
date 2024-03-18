import bp_funcs

from bp_funcs import *

import mpi4py
from mpi4py import MPI

import sys
import os

N_OMP = N_MKL = N_OPENBLAS = 1

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'

L = int(sys.argv[1])
N_a = int(sys.argv[2])
T = int(sys.argv[3])
mu = int(sys.argv[4])
n_fixed = float(sys.argv[5])

p = n_fixed/N_a**2
n = int(N_a**2*p)

# BP
entropy_bp, marginal, physics_density, max_dm, messages, _ = bp_2DLJ_model(
    T, L, N_a,
    chemical_potential=mu,
    cutoff=3.0,
    epsilon=1.0,
    sigma=1.0,
    uv=False,
    gpu=False,
    cyclic=True,
    progbar=True,
    density_compute=True,
    converged = False,
    tol = 1e-5,
    max_iterations=1500,
    uniform=False,
    count = 0,
    damping = True,
    damping_eta = 4e-1,
    reuse = False,
    smudge_factor = 1e-3,
)


sliced_BP_partition_dict = slice_BP_partition_func_dict(
    L, N_a, T, mu, p,
    tol=1e-5,
    max_iterations=1500,
    uniform=False,
    damping=True,
    damping_eta=4e-1,
    smudge_factor=1e-12,
    progbar=True,
    messages_cache=None,
    adjacent=False,
    parallel=False,
)

finite_diff_density = compute_density_from_finite_diff_BP(L, N_a, T, mu)

file = open(f'./Results/2D_BP_L={L}.txt','a')

density, bp_density = slice_BP_density(sliced_BP_partition_dict)
print(f'\nT={T}, L={L}, N_a={N_a}')
print(f'\nBP physics_density = {physics_density}',)
print(f'BP sliced inds density = {density*N_a**2/L**2}')
print(f'BP sliced averaged surrounding density = {bp_density*N_a**2/L**2}')
print(f'BP sliced density naive mean = {(density+bp_density)*N_a**2/L**2/2}')
print(f'BP weighted density = {(density*n + bp_density*(N_a**2-n))/L**2}')
print(f'Finite difference density = {finite_diff_density}')

# Write to file
file.write(f'\nT={T}, L={L}, N_a={N_a}, n_fixed={p*N_a**2}')
file.write(f'\nBP physics_density = {physics_density}')
file.write(f'\nBP sliced inds density = {density*N_a**2/L**2}')
file.write(f'\nBP sliced averaged surrounding density = {bp_density*N_a**2/L**2}')
file.write(f'\nBP sliced density naive mean = {(density+bp_density)*N_a**2/L**2/2}')
file.write(f'\nBP weighted density = {(density*n + bp_density*(N_a**2-n))/L**2}')
file.write(f'\nFinite difference density = {finite_diff_density}')