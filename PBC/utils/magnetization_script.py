import os
import numpy as np

N_OMP = N_MKL = N_OPENBLAS = 1

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'

# T_list = np.linspace(0.1, 5, 30)
T_min = 0.1
T_max = 5
dT = 0.1

if __name__ == "__main__":
    print(f"Running for magnetization")
    os.system(f"mpirun -np 20 python magnetization.py {T_min} {T_max} {dT}")