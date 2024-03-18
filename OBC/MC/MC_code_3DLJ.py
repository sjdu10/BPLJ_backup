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
a = 0.9  # Lattice constant of the underlying simple cubic lattice (in reduced units)
N_a = 3 # Number of lattice points along each direction
L = (N_a-1)*a  # Lattice size
N = N_a**3  # Total number of lattice
T = float(sys.argv[1])  # Temperature
sample_size = int(2e3)  # Total number of Monte Carlo steps

cutoff = 3.0  # Cutoff distance for LJ potential
epsilon = 1.0  # Depth of the potential well
sigma = 1.0  # Length scale in LJ potential, also the distance at which the potential becomes zero


# Lennard-Jones potential function
def V_ij(ri, rj, epsilon=epsilon, sigma=sigma, cutoff=cutoff,uv_cutoff=0.90):
    r_ij = np.linalg.norm(np.array(ri) - np.array(rj))
    if r_ij > cutoff:
        return 0.0
    if r_ij < uv_cutoff:
        r_ij = uv_cutoff
    return 4 * epsilon * ((sigma / r_ij)**12 - (sigma / r_ij)**6)

def compute_h_field(lattice, L=L, N_a=N_a, cutoff=cutoff):
    h_field = np.zeros_like(lattice, dtype=float)
    a = L / (N_a-1)  # Lattice constant of the underlying simple cubic lattice (in reduced units) 
    for i, j, k in itertools.product(range(N_a), range(N_a), range(N_a)):
        count = 0
        for dx in range(-int(cutoff // a), int(cutoff // a) + 1):
            for dy in range(-int(cutoff // a), int(cutoff // a) + 1):
                for dz in range(-int(cutoff // a), int(cutoff // a) + 1):

                    if dx == dy == dz == 0:
                        continue

                    x, y, z = i + dx, j + dy, k + dz

                    if 0 <= x < N_a and 0 <= y < N_a and 0 <= z < N_a: # Check if the neighbor is within the lattice
                        
                        coo_a = np.array([i*a, j*a, k*a])
                        coo_b = np.array([x*a, y*a, z*a])

                        if np.linalg.norm(coo_a - coo_b) >= cutoff:
                            continue

                        h_field[i, j, k] += (0.25) * V_ij(coo_a, coo_b,epsilon=1.0)

                        count += 1

    return h_field

# Function to compute total energy of a configuration with open boundary conditions
def compute_energy(lattice, h_field, L=L, N_a=N_a, cutoff=cutoff):
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

                    # only count each pair once
                    # edge = set([(i, j, k), (x, y, z)])
                    # if edge in edges:
                    #     continue
                    # edges.append(edge)
                    
                    if dy < 0:
                        continue
                    if dy==0 and dx>0:
                        continue
                    if dy==0 and dx==0 and dz>0:
                        continue

                    E += 0.25 * V_ij(coo_a, coo_b) * lattice[i, j, k] * lattice[x, y, z]
    # Add the on-site field contribution to the total energy
    E += np.sum(h_field * lattice)
    # print(f'compute_energy took {time.time()-t0} seconds')
    return E



if __name__ == "__main__":

    lattice = np.random.choice([-1, 1], size=(N_a, N_a, N_a))
    a = L / (N_a-1)  # Lattice constant of the underlying simple cubic lattice (in reduced units)

    h_field = compute_h_field(lattice)

    # Master process
    if rank == 0:
        collected_samples = 0
        E_sum = 0
        M_sum = 0
        pg = Progbar(total=sample_size)
        while collected_samples < sample_size:
            # Receive energy from a worker
            E_received, M_received = comm.recv(source=MPI.ANY_SOURCE)
            E_sum += E_received
            M_sum += M_received
            collected_samples += 1
            pg.update()

            # Check if enough samples have been collected
            if collected_samples >= sample_size:
                terminate = True
                # Send termination signal to all workers
                for i in range(1, size):
                    comm.send(terminate, dest=i)
        
        print(f'T={T}',file=open(f"./3DLJ_L={L}_N_a={N_a}_results.txt", "a"))

        # Compute energy expectation value per site
        E_avg = E_sum / sample_size
        print(f"Energy Expectation Value: {E_avg}", file=open(f"./3DLJ_L={L}_N_a={N_a}_results.txt", "a"))

        # Compute total magnetization expectation value
        M_avg = M_sum / sample_size
        print(f"Total Magnetization Expectation Value: {M_avg}",file=open(f"./3DLJ_L={L}_N_a={N_a}_results.txt", "a"))


    # Worker processes
    else:
        np.random.seed(rank)  # Different seed for each worker
        lattice = np.random.choice([-1, 1], size=(N_a, N_a, N_a))
        terminate = False

        while not terminate:
            for i in range(N_a):
                for j in range(N_a):
                    for k in range(N_a):
                        # Compute energy change if this spin is flipped
                        dE = 0
                        # Only need to consider spins within the cutoff distance
                        for dx in range(-int(cutoff // a), int(cutoff // a) + 1):
                            for dy in range(-int(cutoff // a), int(cutoff // a) + 1):
                                for dz in range(-int(cutoff // a), int(cutoff // a) + 1):
                                    if dx == dy == dz == 0:
                                        continue

                                    x, y, z = i + dx, j + dy, k + dz

                                    if 0 <= x < N_a and 0 <= y < N_a and 0 <= z < N_a: # Check if the neighbor is within the lattice
                                        
                                        coo_a = np.array([i*a, j*a, k*a])
                                        coo_b = np.array([x*a, y*a, z*a])

                                        if np.linalg.norm(coo_a - coo_b) >= cutoff:
                                            continue

                                        dE += -2 * 0.25 * V_ij(coo_a, coo_b) * lattice[i, j, k] * lattice[x, y, z]
                        # Add the on-site field contribution to the total energy
                        dE += -2 * h_field[i, j, k] * lattice[i, j, k]

                        # Metropolis-Hastings algorithm
                        if dE < 0 or np.random.rand() <= np.exp(-dE / T):
                            # Accept the new configuration
                            lattice[i, j, k] *= -1

            # Compute and send energy to master
            E_current = compute_energy(lattice, h_field=h_field)/N
            comm.send((E_current,np.abs(np.sum(lattice))/N), dest=0)

            # Check for termination signal from master
            status = MPI.Status()
            if comm.Iprobe(source=0, status=status):
                terminate = comm.recv(source=0)
