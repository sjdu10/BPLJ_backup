import os
import numpy as np

N_OMP = N_MKL = N_OPENBLAS = 1

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'

# T_list = np.linspace(0.1, 5, 30)
dT = 0.8
T_list = np.arange(0.1, 5, dT)

if __name__ == "__main__":
    for T in T_list:
        print(f"Running for T = {T}")
        os.system(f"mpiexec -np 20 python MC_code_2DLJ.py {T}")