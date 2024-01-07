from Grid import grid
from Basis import basis
from TISE import tise
from Laser import laser
from Hamiltonian import hamiltonian
from Psi import psi
from Propagator import propagator
from Module import *

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import json
import time
import h5py
import os
import gc
from scipy.interpolate import BSpline

import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import sys

comm = PETSc.COMM_WORLD

if __name__ == "__main__":

    if comm.rank == 0:
        start = time.time()

    GRID = True
    BASIS = True
    TISE = True
    LASER = True
    PSI = True
    HAMILTONIAN = True
    PROPAGATE = True

    LOG_PROP = True

    if comm.rank == 0:
        if not os.path.exists("matrix_files"):
            os.mkdir("matrix_files")
        if not os.path.exists("images"):
            os.mkdir("images")
        if not os.path.exists("basis"):
            os.mkdir("basis")
        

    if GRID:
        if comm.rank == 0:
            gridstart = time.time()

        gridInstance = grid()
        gridInstance.print(False)

        if comm.rank == 0:
            gridend = time.time()
            print("Total Time to Construt Grid:",gridend-gridstart)


    if BASIS:
        if comm.rank == 0:
            basisstart = time.time()

        basisInstance = basis()
        basisInstance.createBasis(gridInstance)
        basisInstance.saveBasis(gridInstance,plot = False)
        

        if comm.rank == 0:
            basisend = time.time()
            print("Total Time to Construct Basis:",basisend-basisstart)


    if TISE:
        if comm.rank == 0:
            tisestart = time.time()
    
        tiseInstance = tise()


        # If we dont have the bound states, nor do we have the total matrices then we need to run all of this
        if not (os.path.exists("Hydrogen.h5") and os.path.exists("matrix_files/H_0.bin") and os.path.exists("matrix_files/overlap.bin")):
            tiseInstance.createAllH(basisInstance)
            tiseInstance.createS_R(basisInstance)
            tiseInstance.solveEigensystem()

        if comm.rank == 0:
            tiseend = time.time()
            print("Total Time to Find Eigensystem:",tiseend-tisestart)


    if LASER:
        if comm.rank == 0:
            laserstart = time.time()
    
        laserInstance = laser()
        laserInstance.createPulse(gridInstance)
        laserInstance.plotPulse(True)

        if comm.rank == 0:
            laserend = time.time()
            print("Total Time to Create Laser:",laserend-laserstart)


    if PSI:

        if comm.rank == 0:
            psistart = time.time()
        psiInstance = psi()
        psiInstance.createInitial(basisInstance)
        if comm.rank == 0:
            psiend = time.time()
            print("Total Time to Create Initial Psi:",psiend-psistart)


    if HAMILTONIAN:

        if comm.rank == 0:
            hamstart = time.time()
        
        hamiltonianInstance = hamiltonian()
        hamiltonianInstance.H_MIX(basisInstance,gridInstance)
        hamiltonianInstance.H_ANG(basisInstance,gridInstance)
        hamiltonianInstance.H_ATOM(tiseInstance,basisInstance,gridInstance)
        hamiltonianInstance.S(tiseInstance,basisInstance)
        hamiltonianInstance.PartialAtomic(gridInstance)
        hamiltonianInstance.PartialAngular(gridInstance)
        
        if comm.rank == 0:
            hamend = time.time()
            print("Total Time to Create Interaction:",hamend-hamstart)

    if LOG_PROP:
        PETSc.Log.begin()
        
    if PROPAGATE:

        if comm.rank == 0:
            propstart = time.time()

        propagatorInstance = propagator(tol = 1E-10)
        propagatorInstance.propagateCN(gridInstance,psiInstance,laserInstance,hamiltonianInstance)

        if comm.rank == 0:
            propend = time.time()
            print("Total Time to Propagate:",propend-propstart)
    
    if LOG_PROP:
        PETSc.Log.view()
    
    if comm.rank == 0:
        end = time.time()
        print("Total Simulation Time:",end-start)


        
        

        