import quimb as qu
import quimb.tensor as qtn
import numpy as np
import tnmpa.solvers.quimb_vbp as qbp
import itertools
import concurrent.futures
from quimb.utils import progbar as Progbar

def bp_2DLJ_model(
        T, L, N_a,
        chemical_potential=0,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        uv=False,
        gpu=False,
        cyclic=True,
        progbar=True,
        density_compute=True,
        converged = False,
        tol = 1e-5,
        max_iterations=1500,
        uniform=False,
        count = 0,
        damping = True,
        damping_eta = 4e-1,
        reuse = False,
        smudge_factor = 1e-3,
):
    beta = 1/T
    tn = qtn.tensor_builder.HTN2D_classical_LennardJones_partition_function(
        Lx=L,Ly=L,
        beta=beta,
        Nx=N_a,Ny=N_a,
        cutoff=cutoff,
        epsilon=epsilon,
        sigma=sigma,
        uv_cutoff=uv,
        chemical_potential=chemical_potential,
        cyclic=cyclic,
        )
    if gpu:
        import torch
        tn.apply_to_arrays(lambda x: torch.from_numpy(x).cuda())
        
    # BP
    messages_cache = None
    max_dm = 0

    while not converged:

        if count > 0:
            damping_eta += 5e-2
    
        print('Successfully built the HTN.')

        messages, converged, max_dm = qbp.run_belief_propagation(
        tn, 
        tol=tol,
        max_iterations=max_iterations,
        messages=messages_cache,
        progbar=progbar,
        thread_pool=8,
        uniform=uniform,
        damping=damping,
        eta=damping_eta,
        show_max_dm=True,
        smudge_factor=smudge_factor,
        )

        if reuse:
            messages_cache = messages

        entropy_bp = qbp.compute_free_entropy_from_messages(tn, messages)

        density=0
        if density_compute:
            marginal = qbp.compute_all_index_marginals_from_messages(tn, messages)
            key_format='s{},{}'
            # Compute the density
            density = 0
            for i in range(N_a):
                for j in range(N_a):
                    key = key_format.format(i,j)
                    density += marginal[key][1]
            density /= L**2
        count += 1
        if count>0:
            break
    
    if gpu:
        entropy_bp = float(entropy_bp.cpu().numpy())
        density = float(density.cpu().numpy())
        tn.apply_to_arrays(lambda x: x.cpu().numpy())
        for key in messages:
            messages[key] = messages[key].cpu().numpy()
        for key in marginal:
            marginal[key] = marginal[key].cpu().numpy()
    
    return entropy_bp, marginal, density, max_dm, messages, tn




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



def fix_ind(TN,ind,ind_value):
    """
        Fix the value of an index in a tensor network. Return the modified tensor network with corresponding index dimension reduced to 1.
    """
    tn = TN.copy()
    tn_no_list = list(tn.ind_map[(ind)])
    for tn_no in tn_no_list:
        temp_ind_list = list(tn.tensors[tn_no].inds)
        ind_pos = temp_ind_list.index(ind)
        inds = list(tn.tensors[tn_no].inds)
        inds.pop(ind_pos) # Remove the fixed index
        shape = list(tn.tensors[tn_no].shape)
        shape.pop(ind_pos) # Remove the dimension of the fixed index
        data = tn.tensors[tn_no].data
        # Create a list of slices
        slices = [slice(None)] * data.ndim  # Start with all slices set to ':'
        slices[ind_pos] = slice(ind_value, ind_value + 1)  # Fix the i-th dimension
        new_data = data[tuple(slices)].reshape(shape) # Reduce the dimension of the tensor data
        tn.tensors[tn_no].modify(data=new_data,inds=tuple(inds))
    return tn

def fix_ind_quimb(tn, ind, ind_value):
    tn_config = tn.copy()
    # remove index
    tn_config.isel_({ind: ind_value})
    return tn_config

def fix_inds_quimb(tn, ind_list, ind_value_list):
    tn_config = tn.copy()
    for ind, ind_value in zip(ind_list, ind_value_list):
        tn_config = fix_ind_quimb(tn_config, ind, ind_value)
    return tn_config

def generate_n_bit_strings(n):
    return [list(bits) for bits in itertools.product([0, 1], repeat=n)]

def process_n_bit_string(n_bit_string, tn, sampled_inds, tol, max_iterations, messages_cache, uniform, damping, damping_eta, smudge_factor):
    tn_config = fix_inds_quimb(tn, sampled_inds, n_bit_string)
    # Run BP
    messages, converged, max_dm = qbp.run_belief_propagation(
        tn_config, 
        tol=tol,
        max_iterations=max_iterations,
        messages=messages_cache,
        progbar=False,
        thread_pool=8,
        uniform=uniform,
        damping=damping,
        eta=damping_eta,
        show_max_dm=True,
        smudge_factor=smudge_factor,
    )
    # Compute the partition function
    Z_sliced = np.exp(qbp.compute_free_entropy_from_messages(tn_config, messages))
    marginal = qbp.compute_all_index_marginals_from_messages(tn_config, messages)
    bp_density = 0
    for key,value in marginal.items():
        bp_density += value[1]
    bp_density /= len(list(marginal.keys()))
    return n_bit_string, Z_sliced, max_dm, bp_density

def slice_BP_partition_func_dict(
        L, N_a, T, mu, p, ind_id="s{},{}",
        tol=1e-5,
        max_iterations=1500,
        uniform=False,
        damping=True,
        damping_eta=4e-1,
        smudge_factor=1e-12,
        progbar=True,
        messages_cache=None,
        adjacent=False,
        ):
    """
        Randomly select p-ratio of the sites and fix their index values to manually
        compute the density.
    """
    import itertools
    import random
    tn = qtn.tensor_builder.HTN2D_classical_LennardJones_partition_function(
        Lx=L,Ly=L,
        beta=1/T,
        Nx=N_a,Ny=N_a,
        chemical_potential=mu,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        cyclic=True,
    )

    n = int(N_a**2*p)
    # Randomly select p-ratio of the sites and fixe their index value to manually
    ind_list = []
    for i,j in itertools.product(range(N_a),range(N_a)):
        ind_list.append(ind_id.format(i,j))
    if not adjacent:
        sampled_inds = random.sample(ind_list,n)
    else:
        sampled_inds = []
        for ind in ind_list[:n]:
            sampled_inds.append(ind)

    n_bit_string_list = generate_n_bit_strings(n)

    sliced_BP_partition_dict = {}

    total_Z = 0
    pg = Progbar(total=len(n_bit_string_list))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_n_bit_string, n_bit_string, tn, sampled_inds, tol, max_iterations, messages_cache, uniform, damping, damping_eta, smudge_factor) for n_bit_string in n_bit_string_list]

        for future in concurrent.futures.as_completed(futures):
            n_bit_string, Z_sliced, max_dm, bp_density = future.result()
            total_Z += Z_sliced
            sliced_BP_partition_dict[tuple(n_bit_string)] = [Z_sliced, max_dm, 0, bp_density]
            if progbar:
                pg.update()

    for key in sliced_BP_partition_dict:
        sliced_BP_partition_dict[key][2] = sliced_BP_partition_dict[key][0]/total_Z
    
    return sliced_BP_partition_dict

def slice_BP_density(
        sliced_BP_partition_dict,
):
    density_array = np.zeros(len(list(sliced_BP_partition_dict.keys())[0]))
    density = 0
    bp_density = 0
    p_list = []
    for key,value in sliced_BP_partition_dict.items():
        p = value[2]
        bit_string = np.array(key)
        density_array += p*bit_string
        density += p*np.sum(np.array(key))/len(key)
        bp_density += p*value[3]
        p_list.append(p)
    print(f'Prob norm = {np.sum(p_list)}')
    print(f'On-site Density array = {density_array}')
    print(f'On-site Sliced Density = {density}, On-site surrounding BP density = {bp_density}')
    return density, bp_density



if __name__ == "__main__":
    L = 10
    N_a = 16
    T = 10.0
    mu = 0
    p = 4.1/N_a**2
    n = int(N_a**2*p)

    # BP
    entropy_bp, marginal, physics_density, max_dm, messages, _ = bp_2DLJ_model(
        T, L, N_a,
        chemical_potential=mu,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        uv=False,
        gpu=False,
        cyclic=True,
        progbar=True,
        density_compute=True,
        converged = False,
        tol = 1e-5,
        max_iterations=1500,
        uniform=False,
        count = 0,
        damping = True,
        damping_eta = 4e-1,
        reuse = False,
        smudge_factor = 1e-3,
    )
    

    sliced_BP_partition_dict = slice_BP_partition_func_dict(
        L, N_a, T, mu, p,
        tol=1e-5,
        max_iterations=1500,
        uniform=False,
        damping=True,
        damping_eta=4e-1,
        smudge_factor=1e-12,
        progbar=True,
        messages_cache=None,
        adjacent=True,
    )

    density, bp_density = slice_BP_density(sliced_BP_partition_dict)
    print(f'\nT={T}, L={L}, N_a={N_a}')
    print(f'\nBP physics_density = {physics_density}')
    print(f'BP sliced inds density = {density*N_a**2/L**2}')
    print(f'BP sliced averaged surrounding density = {bp_density*N_a**2/L**2}')
    print(f'BP sliced density naive mean = {(density+bp_density)*N_a**2/L**2/2}')
    print(f'BP weighted density = {(density*n + bp_density*(N_a**2-n))/L**2}')


    
