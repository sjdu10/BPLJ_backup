import quimb as qu
import quimb.tensor as qtn
import numpy as np
import tnmpa.solvers.quimb_vbp as qbp
import sys

L = 8
N_a = 10
N = N_a**3  # Total number of lattice
T = float(sys.argv[1])
beta = 1/T


cutoff = 3.0  # Cutoff distance for LJ potential
epsilon = 1.0  # Depth of the potential well/ Energy unit scale
sigma = 1.0  # Length scale in LJ potential, also the distance at which the potential becomes zero

def contract_HTN_partition(tn):
    """
    Exactly contract the hyper tensor network to get the partition function.
    """
    import quimb.tensor as qtn
    tensor_list = []
    for tensor in tn.tensor_map.values():
        tensor_list.append(tensor)
    value = qtn.tensor_contract(*tensor_list, output_inds=[])
    return value

tn = qtn.tensor_builder.HTN3D_classical_LennardJones_partition_function_spinrep(
    Lx=L,Ly=L,Lz=L,
    beta=beta,
    Nx=N_a,Ny=N_a,Nz=N_a,
    cutoff=cutoff,
    epsilon=epsilon,
    sigma=sigma,
    )

print('Successfully built the HTN.')

# entropy_exact = np.log(contract_HTN_partition(tn))

# BP
converged = False
tol = 1e-10
count = 0

while not converged:
    if count > 0:
        tol *= 10

    messages, converged = qbp.run_belief_propagation(
    tn, 
    tol=tol,
    max_iterations=500, 
    progbar=True,
    thread_pool=8,
    uniform=True,
    damping=True,
    eta=2e-3,
    )
    count += 1
entropy_bp = qbp.compute_free_entropy_from_messages(tn, messages)

print(f'T={T}, Entropy = {entropy_bp}', file=open(f"./3DLJ_L={L}_N_a={N_a}_BP_results.txt", "a"))
# print(f'T={T}, Entropy = {entropy_exact}', file=open(f"./3DLJ_L={L}_N_a={N_a}_exact_results.txt", "a"))
