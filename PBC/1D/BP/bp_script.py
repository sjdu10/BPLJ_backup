import os
import numpy as np

N_OMP = N_MKL = N_OPENBLAS = 1

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'

L = 10
N_a_list = [40,50]
# T_list = np.linspace(0.1, 5, 30)
dT = 1.0
# T_list = np.arange(0.1, 11, dT)
T_list = [3,4,5,6,7,8,9,10]
# chemical_potential_list = np.arange(-1,5,0.5)
chemical_potential_list = [-5]
n_list = [1.0+1e-3,] # number of indices to fix (number of slcies to take)

if __name__ == "__main__":
    for N_a in N_a_list:
        for T in T_list:
            for mu in chemical_potential_list:
                for n in n_list:
                    p = n/N_a**2
                    print(f'Running L = {L}, N_a = {N_a}, T = {T}, mu = {mu}, n = {n}')
                    os.system(f'python3 bp_run.py {L} {N_a} {T} {mu} {n}')
                    print(f'Finished L = {L}, N_a = {N_a}, T = {T}, mu = {mu}, n = {n}')
                    print('-----------------------------------------------')
