from mpi4py import MPI
import numpy as np
import sys
import itertools
from quimb.utils import progbar as Progbar

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize parameters
a = 1.1  # Lattice constant of the underlying simple cubic lattice (in reduced units)
N_a = 5 # Number of lattice points along each direction
L = (N_a-1)*a  # Lattice size
N = N_a**3  # Total number of lattice
T = float(sys.argv[1])  # Temperature

cutoff = 3.0  # Cutoff distance for LJ potential
epsilon = 1.0  # Depth of the potential well
sigma = 1.0  # Length scale in LJ potential, also the distance at which the potential becomes zero

uv = False
random_flip = False 
burn_in = 0
if random_flip:
    sample_size = int(5e4)  # Total number of Monte Carlo steps
else:
    sample_size = int(3e3)  # Total number of Monte Carlo steps

# Lennard-Jones potential function
def V_ij(ri, rj, epsilon=epsilon, sigma=sigma, cutoff=cutoff, uv_cutoff=0.90):
    r_ij = np.linalg.norm(np.array(ri) - np.array(rj))
    if r_ij > cutoff:
        return 0.0
    if uv and r_ij < uv_cutoff:
        r_ij = uv_cutoff
    return 4 * epsilon * ((sigma / r_ij)**12 - (sigma / r_ij)**6)


# Function to compute total energy of a configuration with open boundary conditions
def compute_energy(lattice, L=L, N_a=N_a, cutoff=cutoff):
    # import time
    # t0 = time.time()
    E = 0
    a = L / (N_a-1)  # Lattice constant of the underlying simple cubic lattice (in reduced units) 
    edges = []
    for i, j, k in itertools.product(range(N_a), range(N_a), range(N_a)):
        for dx in range(-int(cutoff // a), int(cutoff // a) + 1):
            for dy in range(-int(cutoff // a), int(cutoff // a) + 1):
                for dz in range(-int(cutoff // a), int(cutoff // a) + 1):

                    if dx == dy == dz == 0:
                        continue

                    x, y, z = i + dx, j + dy, k + dz

                    if (x<0) or (x>=N_a) or (y<0) or (y>=N_a) or (z<0) or (z>=N_a):
                        continue
                        
                    coo_a = np.array([i*a, j*a, k*a])
                    coo_b = np.array([x*a, y*a, z*a])

                    if np.linalg.norm(coo_a - coo_b) >= cutoff:
                        continue

                    if dy < 0:
                        continue
                    if dy==0 and dx>0:
                        continue
                    if dy==0 and dx==0 and dz>0:
                        continue
                
                    E += V_ij(coo_a, coo_b) * lattice[i, j, k] * lattice[x, y, z]

    return E



if __name__ == "__main__":

    if uv:
        file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_cutoff_results.txt", "a")
    else:
        file=open(f"./3DLJ_occ_L={L}_N_a={N_a}_results.txt", "a")

    lattice = np.random.choice([0, 1], size=(N_a, N_a, N_a))
    a = L / (N_a-1)  # Lattice constant of the underlying simple cubic lattice (in reduced units)


    # Master process
    if rank == 0:
        collected_samples = 0
        E_sum = 0
        M_sum = 0
        E_list = []
        pg = Progbar(total=sample_size)
        while collected_samples < sample_size:
            # Receive energy from a worker
            E_received, M_received = comm.recv(source=MPI.ANY_SOURCE)
            E_sum += E_received
            M_sum += M_received
            collected_samples += 1
            pg.update()
            E_list.append(E_received)

            # Check if enough samples have been collected
            if collected_samples >= sample_size:
                terminate = True
                # Send termination signal to all workers
                for i in range(1, size):
                    comm.send(terminate, dest=i)
        
        print(f'T={T}',file=file)

        # Compute energy expectation value per site
        E_avg = E_sum / sample_size
        print(f"Energy Expectation Value: {E_avg}", file=file)

        # Compute total Occupation expectation value
        M_avg = M_sum / sample_size
        print(f"Occupation Number Expectation Value: {M_avg}",file=file)

        # # Store the energy list as a numpy array
        # E_list = np.array(E_list)

        # # Perform blocking analysis
        # block_size = 50
        # num_blocks = int(sample_size/block_size)
        # E_block_avg = np.zeros(num_blocks)
        # for i in range(num_blocks):
        #     E_block_avg[i] = np.mean(E_list[i*block_size:(i+1)*block_size])
        # E_std_between_blocks = np.std(E_block_avg)
        # print(f'Burn-in: {burn_in}',file=file)
        # print(f"Energy Standard Deviation between blocks: {E_std_between_blocks}",file=file)
        # print(f'E_block_avg: {E_block_avg}',file=file)


    # Worker processes
    else:
        np.random.seed(np.random.randint(0, 1000000))  # Different seed for each worker
        lattice = np.random.choice([0, 1], size=(N_a, N_a, N_a))
        terminate = False

        if not random_flip:
            sample_label = 0
            while not terminate:
                sample_label += 1
                # loop over all lattice sites
                for i in range(N_a):
                    for j in range(N_a):
                        for k in range(N_a):
                            # Compute energy change if this site is flipped
                            dE = 0
                            # Only need to consider sites within the cutoff distance
                            for dx in range(-int(cutoff // a)-1, int(cutoff // a) + 1):
                                for dy in range(-int(cutoff // a)-1, int(cutoff // a) + 1):
                                    for dz in range(-int(cutoff // a)-1, int(cutoff // a) + 1):
                                        if dx == dy == dz == 0:
                                            continue

                                        x, y, z = i + dx, j + dy, k + dz

                                        if (x<0) or (x>=N_a) or (y<0) or (y>=N_a) or (z<0) or (z>=N_a):
                                            continue
                                            
                                        coo_a = np.array([i*a, j*a, k*a])
                                        coo_b = np.array([x*a, y*a, z*a])

                                        if np.linalg.norm(coo_a - coo_b) >= cutoff:
                                            continue

                                        if lattice[x, y, z] == 0:
                                            continue

                                        # lattice[x, y, z] must be 1 to contribute to the energy change

                                        dE += (V_ij(coo_a, coo_b) - 2 * V_ij(coo_a, coo_b) * lattice[i, j, k]) * lattice[x, y, z]
                                        
                            # Metropolis-Hastings algorithm
                            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                                # Accept the new configuration
                                lattice[i, j, k] = 1 - lattice[i, j, k]
                
                if sample_label > burn_in:
                    # Compute and send energy to master
                    E_current = compute_energy(lattice)/N
                    comm.send((E_current,np.abs(np.sum(lattice))/N), dest=0)

                    # Check for termination signal from master
                    status = MPI.Status()
                    if comm.Iprobe(source=0, status=status):
                        terminate = comm.recv(source=0)
        else:
            E_lattice = compute_energy(lattice)
            while not terminate:
                # randomly select a site to flip
                i, j, k = np.random.randint(0, N_a, size=3)
                # Compute energy change if this site is flipped
                dE = 0
                # Only need to consider sites within the cutoff distance
                for dx in range(-int(cutoff // a), int(cutoff // a) + 1):
                    for dy in range(-int(cutoff // a), int(cutoff // a) + 1):
                        for dz in range(-int(cutoff // a), int(cutoff // a) + 1):
                            if dx == dy == dz == 0:
                                continue

                            x, y, z = i + dx, j + dy, k + dz

                            if (x<0) or (x>=N_a) or (y<0) or (y>=N_a) or (z<0) or (z>=N_a):
                                continue
                                
                            coo_a = np.array([i*a, j*a, k*a])
                            coo_b = np.array([x*a, y*a, z*a])

                            if np.linalg.norm(coo_a - coo_b) >= cutoff:
                                continue

                            if lattice[x, y, z] == 0:
                                continue

                            # lattice[x, y, z] must be 1 to contribute to the energy change

                            dE += (V_ij(coo_a, coo_b) - 2 * V_ij(coo_a, coo_b) * lattice[i, j, k]) * lattice[x, y, z]

                            flipped_lattice = lattice.copy()
                            flipped_lattice[i, j, k] = 1 - flipped_lattice[i, j, k]
                            E_flipped = compute_energy(flipped_lattice)
                            print(f'Debug: dE = {dE}, dE1 = {E_flipped-E_lattice}')
                # Metropolis-Hastings algorithm
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    # Accept the new configuration
                    lattice[i, j, k] = 1 - lattice[i, j, k]
                    E_lattice += dE

                # Compute and send energy to master
                E_current = E_lattice/N
                comm.send((E_current,np.abs(np.sum(lattice))/N), dest=0)

                # Check for termination signal from master
                status = MPI.Status()
                if comm.Iprobe(source=0, status=status):
                    terminate = comm.recv(source=0)

            

