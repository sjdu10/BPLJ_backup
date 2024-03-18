import quimb as qu
import quimb.tensor as qtn
import numpy as np
import tnmpa.solvers.quimb_vbp as qbp
import sys




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

def marginals_from_messages(messages):
    """index marginals from messages. I did not make use of the tensor network structure here."""
    marginals = []
    # first half of messages is the factor to index messages
    fac_ind_messages = list(messages.items())[:len(messages)//2]
    ind_messages = dict() # dictionary of ind:list of incoming messages
    list_cache = []
    ind_cache = None
    for pair_message in fac_ind_messages: # pair_message: ((factor,ind),message)
        ind = pair_message[0][1]
        message = pair_message[1]
        # Initial step
        if ind_cache is None:
            ind_cache = ind
            list_cache.append(message)
        else:
            if ind_cache==ind:
                list_cache.append(message)
            else:
                ind_messages[ind_cache] = list_cache.copy()
                # Compute the marginals for this ind_cache
                local_Z = 0
                message_product = np.ones((2))
                for message in ind_messages[ind_cache]:
                    message_product *= np.array(message)
                local_Z = np.sum(message_product)
                marginals.append(message_product[0]/local_Z)
                # Reset
                ind_cache = ind
                list_cache = [message]
                
    # Last ind
    ind_messages[ind_cache] = list_cache.copy()
    # Compute the marginals for last ind_cache
    local_Z = 0
    message_product = np.ones((2))
    for message in ind_messages[ind_cache]:
        message_product *= np.array(message)
    local_Z = np.sum(message_product)
    marginals.append(message_product[0]/local_Z)
    return marginals

def magnetization_LJ_BP(T, L, N_a, cutoff=3.0, epsilon=1.0, sigma=1.0, physics=False):
    """
        Compute the magnetization of the Lennard-Jones model using belief propagation.
    """
    beta = 1/T
    tn = qtn.tensor_builder.HTN3D_classical_LennardJones_partition_function_spinrep(
    Lx=L,Ly=L,Lz=L,
    beta=beta,
    Nx=N_a,Ny=N_a,Nz=N_a,
    cutoff=cutoff,
    epsilon=epsilon,
    sigma=sigma,
    )

    tol = 5e-15

    # BP
    converged = False
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
        uniform=False,
        damping=False,
        eta=1e-2,
        )
        count += 1
    entropy_bp = qbp.compute_free_entropy_from_messages(tn, messages)
    marginals_messages = marginals_from_messages(messages)
    M_from_messages = np.sum(np.ones(len(marginals_messages))-2*np.array(marginals_messages))/N

    marginals = []
    if physics:
        for ind in list(tn.ind_map.keys()):
            print(f'T={T} ind={ind}')
            partial_tn = fix_ind(tn,ind,0)

            # BP
            converged = False
            # tol = 1e-12 # previous 1e-15
            count = 0

            while not converged:
                if count > 0:
                    tol *= 10

                messages, converged = qbp.run_belief_propagation(
                partial_tn, 
                tol=tol,
                max_iterations=500, 
                progbar=True,
                thread_pool=8,
                uniform=False,
                damping=False,
                eta=1e-2,
                )
                count += 1
            partial_entropy_bp = qbp.compute_free_entropy_from_messages(partial_tn, messages)
            marginals.append(np.exp(partial_entropy_bp-entropy_bp))
        
        total_m = 0
        for p in marginals:
            total_m += 1-2*p
        M = total_m/N
    else:
        M = 0
    
    return M, M_from_messages


from mpi4py import MPI
import time
import sys
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

T_min = float(sys.argv[1])
T_max = float(sys.argv[2])
dT = float(sys.argv[3])

if __name__ == "__main__":

    # Parameters
    L = 2
    N_a = 3
    N = N_a**3  # Total number of lattice

    cutoff = 3.0  # Cutoff distance for LJ potential
    epsilon = 1.0  # Depth of the potential well/ Energy scale
    sigma = 1.0  # Length scale in LJ potential, also the distance at which the potential becomes zero

    def f(T):
        # Simulate some computation
        return magnetization_LJ_BP(T=T, L=L, N_a=N_a, cutoff=cutoff, epsilon=epsilon, sigma=sigma, physics=True)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        T_list = list(np.arange(T_min, T_max, dT)) # Your long list of t
        print(T_list)
        results = {}  # To store the results of f(t)
        num_workers = size - 1  # Total number of workers
        num_finished_workers = 0  # Count of finished workers

        # Initial distribution of tasks
        for i in range(1, size):
            if T_list:
                t = T_list.pop()
                comm.send(t, dest=i, tag=1)

        # Collect results and distribute new tasks
        while num_finished_workers < num_workers:
            t_result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            temperature = t_result['T']
            M = t_result['M']
            M_messages = t_result['M_messages']
            
            worker_rank = t_result['rank']

            # Store the result
            results[temperature] = {'M': M, 'M_messages': M_messages}
            print(f'T={temperature},M={M},M_messages={M_messages}\n',file=open(f"./Mag_3DLJ_L={L}_N_a={N_a}_BP_results.txt", "a"))
            # Send a new task to the worker if available
            if T_list:
                t = T_list.pop()
                comm.send(t, dest=worker_rank, tag=1)
            else:
                comm.send(None, dest=worker_rank, tag=0)
                num_finished_workers += 1

    else:
        # Worker process
        while True:
            t = comm.recv(source=0, tag=MPI.ANY_TAG)
            if t is None:
                break  # No more tasks

            M,M_messages = f(t)
            comm.send({'T': t, 'M': M, 'M_messages':M_messages,'rank':rank}, dest=0)
