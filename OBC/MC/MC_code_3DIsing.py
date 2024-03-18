from mpi4py import MPI
import numpy as np
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize parameters
L = 8  # Lattice size
N = L * L * L  # Total number of spins
J = 1.0  # Coupling constant
T = float(sys.argv[1])  # Temperature
sample_size = int(1e4)  # Total number of Monte Carlo steps

# Function to compute energy of a configuration
def compute_energy(lattice):
    E = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                nn_sum = (
                    lattice[(i+1)%L, j, k] + lattice[(i-1)%L, j, k] +
                    lattice[i, (j+1)%L, k] + lattice[i, (j-1)%L, k] +
                    lattice[i, j, (k+1)%L] + lattice[i, j, (k-1)%L]
                )
                E += -J * lattice[i, j, k] * nn_sum
    return E / 2  # Each pair counted twice

if __name__ == "__main__":

    config_list = []  # List of configurations

    # Master process
    if rank == 0:
        collected_samples = 0
        E_sum = 0
        M_sum = 0
        while collected_samples < sample_size:
            # Receive energy from a worker
            E_received, M_received = comm.recv(source=MPI.ANY_SOURCE)
            E_sum += E_received
            M_sum += M_received
            collected_samples += 1

            # Check if enough samples have been collected
            if collected_samples >= sample_size:
                terminate = True
                # Send termination signal to all workers
                for i in range(1, size):
                    comm.send(terminate, dest=i)
        
        print(f'T={T}',file=open("./3DIsing_results.txt", "a"))

        # Compute energy expectation value per site
        E_avg = E_sum / sample_size
        print(f"Energy Expectation Value: {E_avg}", file=open("./3DIsing_results.txt", "a"))

        # Compute total magnetization expectation value
        M_avg = M_sum / sample_size
        print(f"Total Magnetization Expectation Value: {M_avg}",file=open("./3DIsing_results.txt", "a"))


    # Worker processes
    else:
        np.random.seed(rank)  # Different seed for each worker
        lattice = np.random.choice([-1, 1], size=(L, L, L))
        terminate = False
        while not terminate:
            for i in range(L):
                for j in range(L):
                    for k in range(L):
                        # Compute energy change if this spin is flipped
                        nn_sum = (
                            lattice[(i+1)%L, j, k] + lattice[(i-1)%L, j, k] +
                            lattice[i, (j+1)%L, k] + lattice[i, (j-1)%L, k] +
                            lattice[i, j, (k+1)%L] + lattice[i, j, (k-1)%L]
                        )
                        dE = 2 * J * lattice[i, j, k] * nn_sum
                        # Metropolis-Hastings algorithm
                        if dE < 0 or np.random.rand() < np.exp(-dE / T):
                            lattice[i, j, k] *= -1
            
            # Compute and send energy to master
            E_current = compute_energy(lattice)/N
            comm.send((E_current,np.abs(np.sum(lattice))/N), dest=0)

            # Check for termination signal from master
            status = MPI.Status()
            if comm.Iprobe(source=0, status=status):
                terminate = comm.recv(source=0)
