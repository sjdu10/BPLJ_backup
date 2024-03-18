import quimb as qu
import quimb.tensor as qtn
import numpy as np
import tnmpa.solvers.quimb_vbp as qbp
import sys

a=0.9  # Lattice constant of the underlying simple cubic lattice (in reduced units)
N_a = 3
L = a * (N_a - 1)  # Length of the cubic box
N = N_a**3  # Total number of lattice
# T = float(sys.argv[1])
damping_eta = 0
shift = 0


dT = 0.1
T_list = np.arange(0.1, 4.9, dT)

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

message_cache = None
T = T_list[0]

for T in T_list:

    beta = 1/T

    tn = qtn.tensor_builder.HTN3D_classical_LennardJones_partition_function_spinrep(
        Lx=L,Ly=L,Lz=L,
        beta=beta,
        Nx=N_a,Ny=N_a,Nz=N_a,
        cutoff=cutoff,
        epsilon=epsilon,
        sigma=sigma,
        shift=shift,
        )
    
    # for i in range(len(tn.tensors)):
    #     tn.tensors[i].modify(data=tn_new.tensors[i].data)
    # entropy_exact = np.log(contract_HTN_partition(tn))

    print('Successfully built the HTN.')

    # BP
    converged = False
    tol = 1e-8
    count = 0

    # while not converged:
    #     if count > 0:
    #         tol *= 10

    messages, converged, max_dm = qbp.run_belief_propagation(
    tn, 
    tol=tol,
    max_iterations=1000, 
    messages=message_cache,
    progbar=True,
    thread_pool=8,
    uniform=True,
    damping=False,
    eta=damping_eta,
    show_max_dm=True,
    )
    count += 1
    entropy_bp = qbp.compute_free_entropy_from_messages(tn, messages)
    message_cache = messages

    print(f'T={T}, Entropy = {entropy_bp}, Damping={damping_eta}, Converged={converged}, max_dm={max_dm}', file=open(f"./3DLJ_L={L}_N_a={N_a}_BP_results.txt", "a"))
    # print(f'T={T}, Entropy = {entropy_exact}', file=open(f"./3DLJ_L={L}_N_a={N_a}_exact_results.txt", "a"))

