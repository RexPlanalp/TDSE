import os
import sys
import json
import numpy as np
import time

if not os.path.exists("photo_files"):
    os.system("mkdir photo_files")


def divide_array_into_parts(E_min, E_max, n,gamma):
    full_array = np.arange(E_min, E_max + 2*gamma, 2*gamma)
    total_length = len(full_array)
    
    size_per_part = total_length // n
    
    parts = []
    
    for i in range(n):
        start_index = i * size_per_part
        if i == n-1:
            end_index = total_length
        else:
            end_index = (i + 1) * size_per_part
        
        part_array = full_array[start_index:end_index]
        parts.append(part_array)
    
    return parts
def divide_array_into_parts(E_min,E_max,n,gamma):
    
    dx = 2*gamma


    ideal_interval_size = (E_max - E_min) / n


    interval_size = round(ideal_interval_size / dx) * dx

    parts = []
    current_start = E_min

    for _ in range(n):
        
        current_start = round(round(current_start / dx) * dx, 3)
        current_end = current_start + interval_size
        
        if current_end > E_max:
            current_end = E_max
        current_end = round(round(current_end / dx) * dx, 3)
    
        parts.append((current_start, current_end))
    
        
        current_start = current_end
    return parts






if len(sys.argv) != 2:
    print("Error: Requires a single argument")
    sys.exit()

try:
    n = int(sys.argv[1])
except ValueError:
    print("Error: Argument must be an integer")
    sys.exit()

try:
    with open('input.json', 'r') as file:
        input_par = json.load(file)
except FileNotFoundError:
    print("Error: input.json file not found")
    sys.exit()
except json.JSONDecodeError:
    print("Error: input.json is not a valid JSON file")
    sys.exit()

E_min = input_par["E"][0]
E_max = input_par["E"][1]
gamma = input_par["E"][2]

parts = divide_array_into_parts(E_min, E_max, n,gamma)

for i in range(n):
    time.sleep(2)
    os.system(f"sbatch /users/becker/dopl4670/Research/TDSE/Circular_Analysis/Circular_Energy/test.sh {parts[i][0]} {parts[i][1]} {i}")
