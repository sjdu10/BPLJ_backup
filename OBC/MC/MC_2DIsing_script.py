import os
import numpy as np

T_list = np.linspace(4, 7.5, 30)

if __name__ == "__main__":
    for T in T_list:
        print(f"Running for T = {T}")
        os.system(f"mpiexec -np 16 python MC_code_3DIsing.py {T}")