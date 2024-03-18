import os
import numpy as np

N_OMP = N_MKL = N_OPENBLAS = 1

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'

# T_list = np.linspace(0.1, 5, 30)
dT = 1.0
# T_list = np.arange(0.1, 11, dT)
T_list = [3]
# chemical_potential_list = np.arange(-1,5,0.5)
chemical_potential_list = [0]

if __name__ == "__main__":
    for T in T_list:
        for chemical_potential in chemical_potential_list:
            print(f"Running for T = {T}, mu={chemical_potential}")
            os.system(f"mpiexec -np 2 python MC_code_1DLJ_occupation.py {T} {chemical_potential}")