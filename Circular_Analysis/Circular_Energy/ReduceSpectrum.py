import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
with open('input.json', 'r') as file:
    input_par = json.load(file)
PES = input_par["PES"][0]
PAD = input_par["PAD"][0]
n = int(sys.argv[1])

if PES:
    total = []
    for i in range(n):
        total += list(np.load(f"photo_files/PES{i}.npy"))
    total = np.real(np.array(total))
    np.save("PES.npy",total)

if PAD:
    total = None
    for i in range(n):
        if total is None:
            total = np.load(f"photo_files/PAD{i}.npy")
        else:
            total = np.hstack((total,np.load(f"photo_files/PAD{i}.npy")))
    np.save("PAD.npy",total)

os.system("rm -rf photo_files")