import quimb as qu
import quimb.tensor as qtn
import numpy as np
import tnmpa.solvers.quimb_vbp as qbp

tn = qtn.tensor_builder.HTN2D_classical_ising_partition_function(Lx=10,Ly=10,beta=0.7)
tn = qtn.tensor_builder.HTN2D_classical_LennardJones_partition_function_spinrep(Lx=3,Ly=3,beta=0.1)
# can also call KsatInstance.htn(mode='dense') for smaller N
# tn = qtn.HTN_random_ksat(
#     2, 
#     3, 
#     alpha=3.0, 
#     # mode must be dense to have a single positive tensor per clause
#     mode='dense', 
#     seed=42,
# )
# tn.hyperinds_resolve_('tree','clustering')

messages, converged = qbp.run_belief_propagation(
    tn, 
    tol=1e-13,
    max_iterations=1000, 
    progbar=True,
    # you can parallelize but its not super efficient
    thread_pool=False,
    uniform=False,
    damping=True,
)