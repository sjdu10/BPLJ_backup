import os
import numpy as np

# T_list = np.linspace(0.1, 5, 30)
dT = 0.5
T_list = np.arange(0.1,10, dT)

if __name__ == "__main__":
    for T in T_list:
        print(f"Running for T = {T}")
        os.system(f"mpiexec -np 20 python MC_code_3DLJ_occupation.py {T}")