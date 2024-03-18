import quimb as qu
import quimb.tensor as qtn
import numpy as np
import tnmpa.solvers.quimb_vbp as qbp
import itertools
import functools
import math
import concurrent.futures
from quimb.utils import progbar as Progbar

from quimb.core import make_immutable
from quimb.tensor.tensor_core import (
    Tensor,
    TensorNetwork,
)


@functools.lru_cache(128)
def classical_LennardJones_S_matrix(beta, Vij):
    """The interaction term for the classical Lennard-Jones model."""
    S = np.array([
        [1, 1],
        [1, math.exp(-beta * Vij)],
    ])
    make_immutable(S)
    return S


def classical_LennardJones_mu_matrix(beta, mu):
    """The chemical potential term for the classical Lennard-Jones model."""
    M = np.array([1, math.exp(beta * mu)])
    make_immutable(M)
    return M


def HTN2D_classical_LennardJones_partition_function(
        Lx,
        Ly,
        beta,
        Nx,
        Ny,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        uv_cutoff=False,
        chemical_potential=0.0,
        cyclic=True,
        ind_id="s{},{}"
):
    try:
        cyclic_x, cyclic_y = cyclic
    except TypeError:
        cyclic_x = cyclic_y = cyclic

    # Lattice spacing
    ax = Lx / (Nx - 1)
    ay = Ly / (Ny - 1)

    if ax != ay or Nx != Ny:
        raise NotImplementedError

    else:
        a = ax
        N_a = Nx

    def LennardJones_potential(
            ri,
            rj,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            uv_cutoff=uv_cutoff
    ):
        r = np.linalg.norm(np.array(ri) - np.array(rj))

        if uv_cutoff:
            if r < 0.9 * sigma:
                r = 0.9 * sigma

        r_on_sigma = r / sigma

        if r_on_sigma > cutoff:
            return 0.0
        else:
            return 4 * epsilon * ((1 / r_on_sigma)**12 - (1 / r_on_sigma)**6)

    ts = []
    for i, j in itertools.product(range(N_a), range(N_a)):
        for dx in range(-int(cutoff // a), int(cutoff // a) + 1):
            for dy in range(-int(cutoff // a), int(cutoff // a) + 1):

                if dx == dy == 0:
                    continue

                x, y = i + dx, j + dy

                if (x < 0) or (x >= N_a) or (y < 0) or (y >= N_a):
                    if not cyclic:
                        # OBC
                        continue

                coo_a = np.array([i * a, j * a])
                coo_b = np.array([x * a, y * a])

                if np.linalg.norm(coo_a - coo_b) >= cutoff * sigma:
                    continue

                if dy < 0 or (dy == 0 and dx > 0):
                    continue

                node_a, node_b = (i, j), (x % N_a, y % N_a)
                inds = ind_id.format(*node_a), ind_id.format(*node_b)
                Vij = LennardJones_potential(coo_a, coo_b)
                data = classical_LennardJones_S_matrix(beta=beta, Vij=Vij)
                ts.append(Tensor(data, inds=inds))

        inds = ind_id.format(i, j)
        data = classical_LennardJones_mu_matrix(beta=beta,
                                                mu=chemical_potential)
        ts.append(Tensor(data, inds=(inds, )))

    return TensorNetwork(ts)

def HTN1D_classical_LennardJones_partition_function(
        L,
        beta,
        N,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        uv_cutoff=False,
        chemical_potential=0.0,
        cyclic=True,
        ind_id="s{}"
):
    a = L / (N - 1)

    def LennardJones_potential(
            ri,
            rj,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            uv_cutoff=uv_cutoff
    ):
        r = np.linalg.norm(np.array(ri) - np.array(rj))

        if uv_cutoff:
            if r < 0.9 * sigma:
                r = 0.9 * sigma

        r_on_sigma = r / sigma

        if r_on_sigma > cutoff:
            return 0.0
        else:
            return 4 * epsilon * ((1 / r_on_sigma)**12 - (1 / r_on_sigma)**6)

    ts = []
    for i in range(N):
        for dx in range(1, int(cutoff // a) + 1):

            x = i + dx

            if x >= N:
                if not cyclic:
                    # OBC
                    continue

            coo_a = np.array([i * a]) # Physical coordinate of the site
            coo_b = np.array([x * a])

            if np.linalg.norm(coo_a - coo_b) >= cutoff * sigma:
                continue

            node_a, node_b = (i, ), (x % N, )
            inds = ind_id.format(*node_a), ind_id.format(*node_b)
            Vij = LennardJones_potential(coo_a, coo_b)
            data = classical_LennardJones_S_matrix(beta=beta, Vij=Vij)
            ts.append(Tensor(data, inds=inds))

        inds = ind_id.format(i)
        data = classical_LennardJones_mu_matrix(beta=beta,
                                                mu=chemical_potential)
        ts.append(Tensor(data, inds=(inds, )))

    return TensorNetwork(ts)


def bp_2DLJ_model(
    T,
    L,
    N_a,
    chemical_potential=0,
    cutoff=3.0,
    epsilon=1.0,
    sigma=1.0,
    uv=False,
    gpu=False,
    cyclic=True,
    progbar=True,
    density_compute=True,
    tol=1e-5,
    max_iterations=1500,
    uniform=False,
    max_count=0,
    damping=True,
    damping_eta=4e-1,
    reuse=False,
    smudge_factor=1e-3,
):
    """
    Run belief propagation on the 2D Lennard-Jones model.
    
    Args:
    ----------
    T: float, 
        temperature
    L: int, 
        size of the 2D lattice
    N_a: int, 
        number of sites in each direction for discretization
    chemical_potential: float, 
        chemical potential
    cutoff: float, 
        cutoff radius for the Lennard-Jones potential
    epsilon: float, 
        energy scale for the Lennard-Jones potential
    sigma: float, 
        length scale for the Lennard-Jones potential
    uv: bool, 
        whether to use the ultraviolet (short-range) cutoff
    gpu: bool, 
        whether to use the GPU implementation
    cyclic: bool, 
        whether to use periodic boundary conditions
    progbar: bool, 
        whether to display a progress bar
    density_compute: bool, 
        whether to compute the BP density
    tol: float, 
        tolerance for the BP convergence
    max_iterations: int, 
        maximum number of BP messages iterations
    uniform: bool, 
        whether to use uniform initial messages
    max_count: int, 
        maximum number of BP trials
    damping: bool, 
        whether to use damping
    damping_eta: float, 
        value of the damping parameter
    reuse: bool, 
        whether to reuse the unconverged messages of the previous BP trial for the next one
    smudge_factor : float,
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    
    Returns:
    ----------
    entropy_bp: float,
        Bethe free entropy computed from the BP messages
    marginal: dict,
        dictionary containing the BP marginal probabilities (used for computing the density)
    density: float,
        density computed from the BP messages
    max_dm: float,
        maximum value of the BP messages difference, used for checking the convergence
    messages: dict,
        dictionary of the BP messages
    tn: quimb.tensor.tensor_core.TensorNetwork2D,
        the hyper tensor network representing the 2D Lennard-Jones model partition function
    """
    beta = 1 / T
    tn = qtn.tensor_builder.HTN2D_classical_LennardJones_partition_function(
        Lx=L,
        Ly=L,
        beta=beta,
        Nx=N_a,
        Ny=N_a,
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
    converged = False
    count = 0
    while not converged:

        if count > 0:  # If the BP did not converge, increase the damping
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

        density = 0
        if density_compute:
            marginal = qbp.compute_all_index_marginals_from_messages(
                tn, messages)
            key_format = 's{},{}'
            # Compute the density
            density = 0
            for i in range(N_a):
                for j in range(N_a):
                    key = key_format.format(i, j)
                    density += marginal[key][1]
            density /= L**2

        count += 1

        if count > max_count:  # If BP did not converge after max_count tries, break
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

def bp_1DLJ_model(
    T,
    L,
    N,
    chemical_potential=0,
    cutoff=3.0,
    epsilon=1.0,
    sigma=1.0,
    uv=False,
    gpu=False,
    cyclic=True,
    progbar=True,
    density_compute=True,
    tol=1e-5,
    max_iterations=1500,
    uniform=False,
    max_count=0,
    damping=True,
    damping_eta=4e-1,
    reuse=False,
    smudge_factor=1e-3,
):
    """
    Run belief propagation on the 1D Lennard-Jones model.
    
    Args:
    ----------
    T: float, 
        temperature
    L: int, 
        size of the 2D lattice
    N: int, 
        number of sites in each direction for discretization
    chemical_potential: float, 
        chemical potential
    cutoff: float, 
        cutoff radius for the Lennard-Jones potential
    epsilon: float, 
        energy scale for the Lennard-Jones potential
    sigma: float, 
        length scale for the Lennard-Jones potential
    uv: bool, 
        whether to use the ultraviolet (short-range) cutoff
    gpu: bool, 
        whether to use the GPU implementation
    cyclic: bool, 
        whether to use periodic boundary conditions
    progbar: bool, 
        whether to display a progress bar
    density_compute: bool, 
        whether to compute the BP density
    tol: float, 
        tolerance for the BP convergence
    max_iterations: int, 
        maximum number of BP messages iterations
    uniform: bool, 
        whether to use uniform initial messages
    max_count: int, 
        maximum number of BP trials
    damping: bool, 
        whether to use damping
    damping_eta: float, 
        value of the damping parameter
    reuse: bool, 
        whether to reuse the unconverged messages of the previous BP trial for the next one
    smudge_factor : float,
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    
    Returns:
    ----------
    entropy_bp: float,
        Bethe free entropy computed from the BP messages
    marginal: dict,
        dictionary containing the BP marginal probabilities (used for computing the density)
    density: float,
        density computed from the BP messages
    max_dm: float,
        maximum value of the BP messages difference, used for checking the convergence
    messages: dict,
        dictionary of the BP messages
    tn: quimb.tensor.tensor_core.TensorNetwork2D,
        the hyper tensor network representing the 2D Lennard-Jones model partition function
    """
    beta = 1 / T
    tn = qtn.tensor_builder.HTN1D_classical_LennardJones_partition_function(
        L,
        beta,
        N,
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
    converged = False
    count = 0
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
            
            density = 0
            if density_compute:
                marginal = qbp.compute_all_index_marginals_from_messages(
                    tn, messages)
                key_format = 's{}'
                # Compute the density
                density = 0
                for i in range(N):
                    key = key_format.format(i)
                    density += marginal[key][1]
                density /= L
                
            count += 1
            
            if count > max_count:
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
                
            
            


def compute_density_from_finite_diff_BP(L, N_a, T, mu, epsilon=1e-2):
    """
    Compute the density using the central finite difference method of the BP Bethe free entropy w.r.t. the chemical potential.
    
    Args:
    ----------
    L: int, 
        size of the 2D lattice
    N_a: int,
        number of sites in each direction for discretization
    T: float,
        temperature
    mu: float,
        chemical potential
    epsilon: float,
        small number for the chemical potential finite difference
    """

    mu_plus = mu + epsilon
    mu_minus = mu - epsilon
    beta = 1 / T
    entropy_bp_plus, _, density_plus, _, _, _ = bp_2DLJ_model(
        T,
        L,
        N_a,
        chemical_potential=mu_plus,
    )

    entropy_bp_minus, _, density_minus, _, _, _ = bp_2DLJ_model(
        T,
        L,
        N_a,
        chemical_potential=mu_minus,
    )

    density_finite_diff = (entropy_bp_plus -
                           entropy_bp_minus) / (2 * epsilon) / beta / L**2
    print(
        f'entropy_bp_plus = {entropy_bp_plus}, entropy_bp_minus = {entropy_bp_minus}'
    )

    return density_finite_diff, density_plus, density_minus


def contract_HTN_partition(tn):
    """
        Exactly contract the hyper tensor network to get the exact partition function.
        Only works for small systems.
        """
    import quimb.tensor as qtn
    tensor_list = []
    for tensor in tn.tensor_map.values():
        tensor_list.append(tensor)
    value = qtn.tensor_contract(*tensor_list, output_inds=[])
    return value


def fix_ind_quimb(tn, ind, ind_value):
    """
    Fix a single index of the tensor network.
    """
    tn_config = tn.copy()
    # remove index
    tn_config.isel_({ind: ind_value})
    return tn_config


def fix_inds_quimb(tn, ind_list, ind_value_list):
    """
    Fix a list of indices of the tensor network.
    Used for TN slicing.
    """
    tn_config = tn.copy()
    for ind, ind_value in zip(ind_list, ind_value_list):
        tn_config = fix_ind_quimb(tn_config, ind, ind_value)
    return tn_config


def generate_n_bit_strings(n):
    """
    Generate all possible n-bit strings.
    """
    return [list(bits) for bits in itertools.product([0, 1], repeat=n)]


def process_n_bit_string(n_bit_string, tn, sampled_inds, tol, max_iterations,
                         messages_cache, uniform, damping, damping_eta,
                         smudge_factor):
    """
    For a given n-bit string, fix the indices of the tensor network and run BP to compute a sliced partition function.
    """
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
    Z_sliced = np.exp(
        qbp.compute_free_entropy_from_messages(tn_config, messages))
    marginal = qbp.compute_all_index_marginals_from_messages(
        tn_config, messages)
    print(
        f'Sampled inds = {sampled_inds}, Value = {n_bit_string}, Z_sliced = {Z_sliced}'
    )
    print(f'Marginal = {marginal}')
    bp_density = 0
    for key, value in marginal.items():
        bp_density += value[1]
    bp_density /= len(list(marginal.keys()))
    return n_bit_string, Z_sliced, max_dm, bp_density


def slice_BP_partition_func_dict(L,
                                 N_a,
                                 T,
                                 mu,
                                 p,
                                 ind_id="s{},{}",
                                 tol=1e-5,
                                 max_iterations=1500,
                                 uniform=False,
                                 damping=True,
                                 damping_eta=4e-1,
                                 smudge_factor=1e-12,
                                 progbar=True,
                                 messages_cache=None,
                                 adjacent=False,
                                 parallel=True):
    """
        Randomly select p-ratio of the sites and fix their indices values to manually
        compute the density.
        
        Returns:
        ----------
        sliced_BP_partition_dict: dict,
            dictionary containing [sliced partition function, slice BP max_dm, Z_sliced/Z_total, BP density]
        total_Z: float,
            total partition function as the sum of the sliced partition functions
    """
    import itertools
    import random
    tn = qtn.tensor_builder.HTN2D_classical_LennardJones_partition_function(
        Lx=L,
        Ly=L,
        beta=1 / T,
        Nx=N_a,
        Ny=N_a,
        chemical_potential=mu,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        cyclic=True,
    )

    n = int(N_a**2 * p)
    # Randomly select p-ratio of the sites and fixe their index value to manually
    ind_list = []
    for i, j in itertools.product(range(N_a), range(N_a)):
        ind_list.append(ind_id.format(i, j))
    if not adjacent:
        sampled_inds = random.sample(ind_list, n)
    else:
        sampled_inds = []
        for ind in ind_list[:n]:
            sampled_inds.append(ind)

    n_bit_string_list = generate_n_bit_strings(n)

    sliced_BP_partition_dict = {}

    total_Z = 0
    pg = Progbar(total=len(n_bit_string_list))

    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_n_bit_string, n_bit_string, tn,
                                sampled_inds, tol, max_iterations,
                                messages_cache, uniform, damping, damping_eta,
                                smudge_factor)
                for n_bit_string in n_bit_string_list
            ]

            for future in concurrent.futures.as_completed(futures):
                n_bit_string, Z_sliced, max_dm, bp_density = future.result()
                total_Z += Z_sliced
                sliced_BP_partition_dict[tuple(n_bit_string)] = [
                    Z_sliced, max_dm, 0, bp_density
                ]
                if progbar:
                    pg.update()
    else:
        for n_bit_string in n_bit_string_list:
            n_bit_string, Z_sliced, max_dm, bp_density = process_n_bit_string(
                n_bit_string, tn, sampled_inds, tol, max_iterations,
                messages_cache, uniform, damping, damping_eta, smudge_factor)
            total_Z += Z_sliced
            sliced_BP_partition_dict[tuple(n_bit_string)] = [
                Z_sliced, max_dm, 0, bp_density
            ]
            if progbar:
                pg.update()

    for key in sliced_BP_partition_dict:
        sliced_BP_partition_dict[key][
            2] = sliced_BP_partition_dict[key][0] / total_Z

    return sliced_BP_partition_dict, total_Z


def compute_density_from_finite_diff_BP_sliced(L, N_a, T, mu, p, epsilon=1e-2):
    """
    Finite difference calculation of density from the sliced BP partition function.
    """
    mu_plus = mu + epsilon
    mu_minus = mu - epsilon
    beta = 1 / T
    _, Z_plus = slice_BP_partition_func_dict(
        L,
        N_a,
        T,
        mu=mu_plus,
        p=p,
        tol=1e-5,
        max_iterations=1500,
        uniform=False,
        damping=True,
        damping_eta=4e-1,
        smudge_factor=1e-12,
        progbar=True,
        messages_cache=None,
        adjacent=True,
        parallel=False,
    )
    # print('Finished computing Z_plus')
    _, Z_minus = slice_BP_partition_func_dict(
        L,
        N_a,
        T,
        mu=mu_minus,
        p=p,
        tol=1e-5,
        max_iterations=1500,
        uniform=False,
        damping=True,
        damping_eta=4e-1,
        smudge_factor=1e-12,
        progbar=True,
        messages_cache=None,
        adjacent=True,
        parallel=False,
    )
    # print('Finished computing Z_minus')
    entropy_bp_plus = np.log(Z_plus)
    entropy_bp_minus = np.log(Z_minus)
    # print(f'entropy_bp_plus = {entropy_bp_plus}, entropy_bp_minus = {entropy_bp_minus}')

    density_finite_diff = (entropy_bp_plus -
                           entropy_bp_minus) / (2 * epsilon) / beta / L**2

    return density_finite_diff


def slice_BP_density(
    sliced_BP_partition_dict,
    print_density=False,
):
    density_array = np.zeros(len(list(sliced_BP_partition_dict.keys())[0]))
    density = 0
    bp_density = 0
    p_list = []
    for key, value in sliced_BP_partition_dict.items():
        p = value[2]
        bit_string = np.array(key)
        density_array += p * bit_string
        density += p * np.sum(np.array(key)) / len(key)
        bp_density += p * value[3]
        p_list.append(p)
    print(f'Prob norm = {np.sum(p_list)}')
    if print_density:
        print(f'On-site Density array = {density_array}')
        print(
            f'On-site Sliced Density = {density}, On-site surrounding BP density = {bp_density}'
        )
    return density, bp_density


if __name__ == "__main__":
    import sys
    import os

    N_OMP = N_MKL = N_OPENBLAS = 1

    os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
    os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
    os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'

    L = 10
    N_a = 30
    T = 3.0
    mu = -15.0
    n_fixed = float(sys.argv[1])
    p = n_fixed / N_a**2
    n = int(N_a**2 * p)

    # BP
    entropy_bp, marginal, physics_density, max_dm, messages, _ = bp_2DLJ_model(
        T,
        L,
        N_a,
        chemical_potential=mu,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        uv=False,
        gpu=False,
        cyclic=True,
        progbar=True,
        density_compute=True,
        tol=1e-5,
        max_iterations=1500,
        uniform=False,
        damping=True,
        damping_eta=4e-1,
        reuse=False,
        smudge_factor=1e-3,
    )

    sliced_BP_partition_dict, total_Z_scliced = slice_BP_partition_func_dict(
        L,
        N_a,
        T,
        mu,
        p,
        tol=1e-5,
        max_iterations=1500,
        uniform=False,
        damping=True,
        damping_eta=4e-1,
        smudge_factor=1e-12,
        progbar=True,
        messages_cache=None,
        adjacent=True,
        parallel=False,
    )
    # file = open(f'./Results/2D_BP_L={L}.txt','a')
    finite_diff_density = compute_density_from_finite_diff_BP(L, N_a, T, mu)
    finite_diff_density_sliced = compute_density_from_finite_diff_BP_sliced(
        L, N_a, T, mu, p)

    density, bp_density = slice_BP_density(sliced_BP_partition_dict)

    print(f'\nBP partition function = {np.exp(entropy_bp)}')
    print(f'BP sliced total partition function = {total_Z_scliced}')

    print(f'\nT={T}, L={L}, N_a={N_a}')
    print(f'\nBP physics_density = {physics_density}', )
    print(f'BP sliced inds density = {density*N_a**2/L**2}')
    print(f'BP sliced averaged surrounding density = {bp_density*N_a**2/L**2}')
    print(
        f'BP sliced density naive mean = {(density+bp_density)*N_a**2/L**2/2}')
    print(
        f'BP weighted density w.r.t. number of sites (sliced sites n and unsliced sites N_a**2-n) = {(density*n + bp_density*(N_a**2-n))/L**2}'
    )
    print((f'BP Finite difference density = {finite_diff_density}'))
    print((
        f'BP Finite difference density sliced = {finite_diff_density_sliced}'))

    # # Write to file
    # file.write(f'\nT={T}, L={L}, N_a={N_a}, n_fixed={p*N_a**2}')
    # file.write(f'\nBP physics_density = {physics_density}')
    # file.write(f'\nBP sliced inds density = {density*N_a**2/L**2}')
    # file.write(f'\nBP sliced averaged surrounding density = {bp_density*N_a**2/L**2}')
    # file.write(f'\nBP sliced density naive mean = {(density+bp_density)*N_a**2/L**2/2}')
    # file.write(f'\nBP weighted density = {(density*n + bp_density*(N_a**2-n))/L**2}')
