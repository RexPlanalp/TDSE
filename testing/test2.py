from mpi4py import MPI
import numpy as np
import time

# Define the function to integrate
def f(x):
    return np.sin(x)

# Trapezoidal rule for a segment of the integral
def integrate_segment(f, a, b, N):
    h = (b - a) / N
    result = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1, N):
        result += f(a + i * h)
    result *= h
    return result

# Main code
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Interval and number of trapezoids
a = 0.0
b = np.pi * 1000000  # Large interval to ensure enough workload
N = 1000000  # Total number of trapezoids

# Timing starts here
comm.Barrier()  # Ensure all processes start timing at the same point
start_time = time.time()

# Divide the problem among the available MPI processes
local_N = int(N / size)
local_a = a + rank * local_N * (b - a) / N
local_b = local_a + local_N * (b - a) / N

# Each process calculates its segment of the integral
local_integral = integrate_segment(f, local_a, local_b, local_N)

# Sum up the results across all processes
total_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)

# Timing ends here
end_time = time.time()
elapsed_time = end_time - start_time

# Output the result
if rank == 0:
    print(f"Total integral is {total_integral:.6f}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds with {size} processes")

# To run this script, save it as `parallel_integral.py` and execute it with MPI, for example:
# mpiexec -n 4 python parallel_integral.py
