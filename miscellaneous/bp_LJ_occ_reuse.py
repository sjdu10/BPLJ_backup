import quimb as qu
import quimb.tensor as qtn
import numpy as np
import tnmpa.solvers.quimb_vbp as qbp
import sys

a = 0.75  # Lattice constant of the underlying simple cubic lattice (in reduced units)
N_a = 10  # Number of lattice points along one direction
L = a * (N_a - 1)  # Length of the cubic box
N = N_a**3  # Total number of lattice
uv = False  # Whether to use UV cutoff
density_compute = True
# chemical_potential = 5
chemical_potential_list = np.arange(0.5, 10, 5e-1)
reuse = False

dT = 0.1
# T_list = np.arange(0.1, 10, dT)
T_list = [3.0]
# T_list = np.arange(15.0, 10.0, dT)

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

# entropy_exact = np.log(contract_HTN_partition(tn))



for T in T_list:
    for chemical_potential in chemical_potential_list:
        beta = 1/T

        tn = qtn.tensor_builder.HTN3D_classical_LennardJones_partition_function(
            Lx=L,Ly=L,Lz=L,
            beta=beta,
            Nx=N_a,Ny=N_a,Nz=N_a,
            cutoff=cutoff,
            epsilon=epsilon,
            sigma=sigma,
            uv_cutoff=uv,
            chemical_potential=chemical_potential,
            )
        # BP
        converged = False
        tol = 1e-10
        count = 0
        max_dm = 0
        damping_eta = 3e-1
        max_dm_cache = 0
        messages_cache = None

        beta = 1/T
        while not converged:

            if count > 0:
                damping_eta += 5e-2
        
            print('Successfully built the HTN.')

            messages, converged, max_dm = qbp.run_belief_propagation(
            tn, 
            tol=tol,
            max_iterations=1000, 
            messages=messages_cache,
            progbar=True,
            thread_pool=8,
            uniform=False,
            damping=True,
            eta=damping_eta,
            show_max_dm=True,
            )

            count += 1
            if count % 5 == 0:
                tol *= 10


            max_dm_cache = max_dm

            if reuse:
                messages_cache = messages

            entropy_bp = qbp.compute_free_entropy_from_messages(tn, messages)
            density=0
            if density_compute:
                marginal = qbp.compute_all_index_marginals_from_messages(tn, messages)
                key_format='s{},{},{}'
                # Compute the density
                density = 0
                for i in range(N_a):
                    for j in range(N_a):
                        for k in range(N_a):
                            key = key_format.format(i,j,k)
                            density += marginal[key][1]
                density /= L**3
            
            if not converged and count > 3:
                if uv:
                    print(f'T={T}, Entropy = {entropy_bp}, Converged={converged}, damping={damping_eta}, density N/V={density}, chemical potential={chemical_potential}, Not Converged!', file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_cutoff_BP_results.txt", "a"))
                    converged = False
                else:
                    if dT>0:
                        print(f'T={T}, Entropy = {entropy_bp}, Converged={converged}, damping={damping_eta}, density N/V={density}, chemical potential={chemical_potential}, Not Converged!', file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_BP_results.txt", "a"))
                        converged = False
                    else:
                        print(f'T={T}, Entropy = {entropy_bp}, Converged={converged}, damping={damping_eta}, density N/V={density}, chemical potential={chemical_potential}, Not Converged!', file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_BP_results_reverse.txt", "a"))
                        converged = False
                
                break
        

        if not uv:
            if converged:
                if dT>0:
                    print(f'T={T}, Entropy = {entropy_bp}, Converged={converged}, damping={damping_eta}, density N/V={density}, chemical potential={chemical_potential}', file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_BP_results.txt", "a"))
                else:
                    print(f'T={T}, Entropy = {entropy_bp}, Converged={converged}, damping={damping_eta}, density N/V={density}, chemical potential={chemical_potential}', file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_BP_results_reverse.txt", "a"))
        else:
            if converged:
                print(f'T={T}, Entropy = {entropy_bp}, Converged={converged}, damping={damping_eta}, density N/V={density}, chemical potential={chemical_potential}', file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_cutoff_BP_results.txt", "a"))

# print(f'T={T}, Entropy = {entropy_exact}', file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_Exact_results.txt", "a"))
