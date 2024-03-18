import os
import numpy as np

N_OMP = N_MKL = N_OPENBLAS = 1
# # Number of cores used when performing linear algebra operations: N_MPI * N_OMP. Make sure this is less or equal to the number of total threadings.

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'
os.environ['MKL_DOMAIN_NUM_THREADS'] = f'{N_MKL}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'

# T_list = np.linspace(0.1, 5, 30)
dT = 1.0
T_list = np.arange(5, 6, dT)
occ = False
# T_list = [0.1,0.2]

if __name__ == "__main__":
    for T in T_list:
        print(f"Running for T = {T}")
        if occ:
            os.system(f"python bp_LJ_occ.py {T}")
        else:
            os.system(f"python bp_LJ.py {T}")