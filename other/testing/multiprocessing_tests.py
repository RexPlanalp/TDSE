import multiprocessing
from multiprocessing import Pool

def sum_of_squares(n):
    return sum(i**2 for i in range(n))

def parallel_sum_of_squares(numbers):
    num_cores = 8
    
    
    
    with Pool(processes=num_cores) as pool:
        results = pool.map(sum_of_squares, numbers)
    
    return results

if __name__ == "__main__":
    
    numbers = range(1, 10000, 1) 
    
    
    import time
    start_time = time.time()
    
    
    results = parallel_sum_of_squares(numbers)
    
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time} seconds")
