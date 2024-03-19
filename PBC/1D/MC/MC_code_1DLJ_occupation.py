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
# a = 0.65  # Lattice constant of the underlying square lattice (in reduced units)
# N_a = 10  # Number of lattice points along each direction
# L = (N_a - 1) * a  # Lattice size

L = 8
N_a = 50
a = L / (N_a - 1)

N = N_a  # Total number of lattice points in 2D
T = float(sys.argv[1])  # Temperature
cyclic = True
chemical_potential = float(sys.argv[2])

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

# Function to compute total energy of a configuration with open/periodic boundary conditions 
def compute_energy_2D(lattice, L=L, N_a=N_a, cutoff=cutoff,cyclic=cyclic,chemical_potential=chemical_potential):
    E = 0
    a = L / (N_a - 1)
    for i, j in itertools.product(range(N_a), range(N_a)):
        for dx in range(-int(cutoff // a), int(cutoff // a) + 1):
            for dy in range(-int(cutoff // a), int(cutoff // a) + 1):

                if dx == dy == 0:
                    continue

                x, y = i + dx, j + dy

                if (x < 0) or (x >= N_a) or (y < 0) or (y >= N_a):
                    if not cyclic:
                        continue

                coo_a = np.array([i * a, j * a])
                coo_b = np.array([x * a, y * a])

                if np.linalg.norm(coo_a - coo_b) >= cutoff:
                    continue

                if dy < 0 or (dy == 0 and dx > 0):
                    continue

                E += V_ij(coo_a, coo_b) * lattice[i, j] * lattice[x%N_a, y%N_a]
        E += -chemical_potential * lattice[i, j]
    return E

def compute_energy_1D(lattice, L=L, N_a=N_a, cutoff=cutoff,cyclic=cyclic,chemical_potential=chemical_potential):
    E = 0
    a = L / (N_a - 1)
    for i in range(N_a):
        for dx in range(1, int(cutoff // a) + 1):
            x = i + dx
            if x >= N_a:
                if not cyclic:
                    continue
                
            if lattice[i] == 0 or lattice[x%N_a] == 0:
                continue

            coo_a = np.array([i * a])
            coo_b = np.array([x * a])

            if np.linalg.norm(coo_a - coo_b) >= cutoff:
                continue

            E += V_ij(coo_a, coo_b) * lattice[i] * lattice[x%N_a]
        E += -chemical_potential * lattice[i]
    return E

if __name__ == "__main__":
    if uv:
        file = open(f"./Results/1DLJ_occ_L={L}_N_a={N_a}_cutoff_results.txt", "a")
    else:
        file = open(f"./Results/1DLJ_occ_L={L}_N_a={N_a}_results_pbc.txt", "a")
    
    lattice = np.random.choice([0, 1], size=(N_a))
    a = L / (N_a - 1)  # Lattice constant of the underlying square lattice (in reduced units)

    # Master process
    if rank == 0:
        print(f'L={L}, N_a={N_a}, a={a}, cutoff={cutoff}, epsilon={epsilon}, sigma={sigma}, uv={uv}, random_flip={random_flip}, burn_in={burn_in}', file=file)
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

        np.save(f'./Results/1DLJ_occ_L={L}_N_a={N_a}_T={T}_E_list.npy', E_list)
        

        # Compute energy expectation value per site
        E_avg = E_sum / sample_size
        
        # Compute total Occupation expectation value
        M_avg = M_sum / sample_size

        print(f'T={T}, Energy = {E_avg}, Occupation Number = {M_avg}, Density = {M_avg*N/L**2}, chemical potential = {chemical_potential}', file=file)

    # Worker processes
    else:
        # 1D lattice
        np.random.seed(np.random.randint(0, 1000000))
        lattice = np.random.choice([0, 1], size=(N_a))
        terminate = False
        
        while not terminate:
            # loop over all lattice sites
            for i in range(N_a):
                # Compute energy change if this site is flipped
                dE = 0
                # Only need to consider sites within the cutoff distance
                for dx in range(-int(cutoff // a) - 1, int(cutoff // a) + 1):
                    if dx == 0:
                        continue
                    
                    x = i + dx
                    
                    if x >= N_a or x < 0:
                        if not cyclic:
                            continue
                    
                    if lattice[x%N_a] == 0: # lattice[x] must be 1 to contribute to the energy change
                        continue

                    coo_a = np.array([i * a])
                    coo_b = np.array([x * a])

                    if np.linalg.norm(coo_a - coo_b) >= cutoff:
                        continue

                    dE += (1-2*lattice[i]) * V_ij(coo_a, coo_b) * lattice[x%N_a]
                    
                dE += -chemical_potential * lattice[i]
                
                # Metropolis-Hastings algorithm
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    # Accept the new configuration
                    lattice[i] = 1 - lattice[i]
        

            # Compute and send energy to master
            E_current = compute_energy_1D(lattice) / N
            comm.send((E_current, np.abs(np.sum(lattice)) / N), dest=0)

            # Check for termination signal from master
            status = MPI.Status()
            if comm.Iprobe(source=0, status=status):
                terminate = comm.recv(source=0)
