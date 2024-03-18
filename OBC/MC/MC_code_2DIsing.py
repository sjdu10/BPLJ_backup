from mpi4py import MPI
import numpy as np
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize parameters
L = int(30)  # Lattice size
N = L * L  # Total number of spins
J = 1.0  # Coupling constant
T = float(sys.argv[1])  # Temperature
# T = 2.5
sample_size = int(1e4)  # Total number of Monte Carlo steps
progress_bar_length = 50  # Length of the progress bar

# Function to compute energy of a configuration
def compute_energy(lattice):
    # Periodic boundary conditions
    E = 0
    for i in range(L):
        for j in range(L):
            nn_sum = (
                lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
            )
            E += -J * lattice[i, j] * nn_sum
    return E / 2  # Each pair counted twice


# Function to compute energy of a configuration with open boundary conditions
def compute_energy_obc(lattice):
    E = 0
    rows, cols = lattice.shape
    for i in range(rows):
        for j in range(cols):
            nn_sum = 0
            if i > 0:
                nn_sum += lattice[i-1, j]
            if i < rows - 1:
                nn_sum += lattice[i+1, j]
            if j > 0:
                nn_sum += lattice[i, j-1]
            if j < cols - 1:
                nn_sum += lattice[i, j+1]
            E += -J * lattice[i, j] * nn_sum
    return E / 2  # Each pair counted twice


# # Function to update and display progress bar
# def update_progress(progress):
#     bar = "=" * int(progress_bar_length * progress)
#     print(f"\rProgress: [{bar:{progress_bar_length}s}] {progress*100:.2f}%", end="", flush=True)



if __name__ == "__main__":
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

            # # Update progress bar
            # update_progress(collected_samples / sample_size)

            # Check if enough samples have been collected
            if collected_samples >= sample_size:
                # Send termination signal to all workers
                terminate = True
                for i in range(1, size):
                    comm.send(terminate, dest=i)
        
        print("\nSample collection complete.")
        print(f"T={T}", file=open("./results_obc.txt", "a"))

        # Compute energy expectation value
        E_avg = E_sum / sample_size
        print(f"Energy Expectation Value: {E_avg}",file=open("./results_obc.txt", "a"))

        # Compute total magnetization expectation value
        M_avg = M_sum / sample_size
        print(f"Total Magnetization Expectation Value: {M_avg}", file=open("./results_obc.txt", "a"))


    # Worker processes
    else:
        np.random.seed(rank)  # Different seed for each worker
        lattice = np.random.choice([-1, 1], size=(L, L))
        terminate = False
        while not terminate:
            # Loop over all spins
            for i in range(L):
                for j in range(L):
                    # Compute energy change if this spin is flipped
                    nn_sum = (
                        lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                        lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
                    )
                    dE = 2 * J * lattice[i, j] * nn_sum
                    
                    # Metropolis-Hastings algorithm
                    if dE < 0 or np.random.rand() < np.exp(-dE / T):
                        lattice[i, j] *= -1
            
            # Compute and send energy to master
            E_current = compute_energy_obc(lattice)/N # energy per site
            comm.send((E_current, np.abs(np.sum(lattice))/N), dest=0)

            # Check for termination signal from master
            status = MPI.Status()
            if comm.Iprobe(source=0, status=status):
                terminate = comm.recv(source=0)
        
