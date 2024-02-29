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
    with open('input.json', 'r') as file:
            input_par = json.load(file)

    if comm.rank == 0:
        start = time.time()

    GRID = True
    BASIS = True
    TISE = True
    LASER = True
    PSI = True
    HAMILTONIAN = True
    PROPAGATE = True

    LOG_PROP = False
    DETAILS = True

    PRINT_GRID = False
    PLOT_BASIS = False
    SAVE_BASIS = False
    PLOT_PULSE = False
    PROP_TOL = 1E-10

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

        if comm.rank == 0 and DETAILS:
            print("##########################################################")

        if comm.rank == 0 and DETAILS:
            print("Initizializing Grid Instance")
            print("Creating Grid")
        gridInstance = grid()

        if comm.rank == 0 and DETAILS:
            print(f"Printing Grid: {PRINT_GRID}")
        gridInstance.print(False)

        if comm.rank == 0:
            gridend = time.time()
            print("Total Time to Construt Grid:",round(gridend-gridstart,4),"seconds")
        
        if comm.rank == 0 and DETAILS:
            print("##########################################################")


    if BASIS:
        if comm.rank == 0:
            basisstart = time.time()

        if comm.rank == 0 and DETAILS:
            print("Initializing Basis Instance")
        basisInstance = basis()
        if comm.rank == 0 and DETAILS:
            print("Constructing Basis")
        basisInstance.createBasis(gridInstance)
        if comm.rank == 0 and DETAILS:
            print(f"Plotting Basis: {PLOT_BASIS}")
            print(f"Saving Basis: {SAVE_BASIS}")
        basisInstance.saveBasis(gridInstance,plot = PLOT_BASIS,save = SAVE_BASIS)
        

        if comm.rank == 0:
            basisend = time.time()
            print("Total Time to Construct Basis:",round(basisend-basisstart,4),"seconds")

        if comm.rank == 0 and DETAILS:
            print("##########################################################")


    if TISE:
        if comm.rank == 0:
            tisestart = time.time()

        if comm.rank == 0 and DETAILS:
            print("Initzializing Grid Instance")
        tiseInstance = tise()


        
        if not (os.path.exists("Hydrogen.h5") and os.path.exists("matrix_files/H_0.bin") and os.path.exists("matrix_files/overlap.bin")):
            

            if comm.rank == 0 and DETAILS:
                print("Constructing Atomic Hamiltonians")
            tiseInstance.createAllH(basisInstance)

            if comm.rank == 0 and DETAILS:
                print("Constructing Overlap Matrix")
            tiseInstance.createS_R(basisInstance)

            if comm.rank ==0 and DETAILS:
                print("Solving Eigensystem")
            tiseInstance.solveEigensystem()

        if comm.rank == 0:
            tiseend = time.time()
            print("Total Time to Find Eigensystem:",round(tiseend-tisestart,4),"seconds")
        if comm.rank == 0 and DETAILS:
            print("##########################################################")


    if LASER:
        if comm.rank == 0:
            laserstart = time.time()

        if comm.rank == 0 and DETAILS:
            print("Initializing Laser Instance")
        laserInstance = laser()
        if comm.rank == 0 and DETAILS:
            print("Constructing Laser Pulse")
        laserInstance.createEnvelope()
        laserInstance.createCarrier()
        laserInstance.createAmplitude()
        laserInstance.createPulse(gridInstance)
        if comm.rank == 0 and DETAILS:
            print(f"Plotting Pulse: {PLOT_PULSE}")
        laserInstance.plotPulse(True)

        if comm.rank == 0:
            laserend = time.time()
            print("Total Time to Create Laser:",round(laserend-laserstart,4),"seconds")
        if comm.rank == 0 and DETAILS:
            print("##########################################################")


    if PSI:

        if comm.rank == 0:
            psistart = time.time()
        
        if comm.rank == 0 and DETAILS:
            print("Initializing Psi Instance")
        psiInstance = psi()
        if comm.rank == 0 and DETAILS:
            print("Constructing Initial Psi")
        psiInstance.createInitial(basisInstance)
        if comm.rank == 0:
            psiend = time.time()
            print("Total Time to Create Initial Psi:",round(psiend-psistart,4),"seconds")

        if comm.rank == 0 and DETAILS:
            print("##########################################################")


    if HAMILTONIAN:

        if comm.rank == 0:
            hamstart = time.time()
        
        if comm.rank == 0 and DETAILS:
            print("Initializing Hamiltonian Instance")
        hamiltonianInstance = hamiltonian()


        if comm.rank == 0 and DETAILS:
            print("Constructing Propagation Hamiltonians")
        if laserInstance.gauge == "velocity":
            hamiltonianInstance.H_MIX(basisInstance,gridInstance,tiseInstance)
            hamiltonianInstance.H_ANG(basisInstance,gridInstance,tiseInstance)
            hamiltonianInstance.PartialAngularVelocity(gridInstance)
        elif laserInstance.gauge == "length":
            hamiltonianInstance.H_LENGTH(basisInstance,gridInstance)
            hamiltonianInstance.PartialAngularLength(gridInstance)

        if comm.rank == 0 and DETAILS:
            print("Constructing Total Atomic and Total Overlap Matrices")
        hamiltonianInstance.H_ATOM(tiseInstance,basisInstance,gridInstance)
        hamiltonianInstance.S(tiseInstance,basisInstance)
        hamiltonianInstance.PartialAtomic(gridInstance)
        
        
        if comm.rank == 0:
            hamend = time.time()
            print("Total Time to Create Interaction:",round(hamend-hamstart,4),"seconds")
        
        if comm.rank == 0 and DETAILS:
            print("##########################################################")

    if LOG_PROP:
        PETSc.Log.begin()

    if PROPAGATE:

        if comm.rank == 0:
            propstart = time.time()

        if comm.rank == 0 and DETAILS:
            print("Initializing Propagator Instance")
        propagatorInstance = propagator(tol = PROP_TOL)

        if comm.rank == 0 and DETAILS:
            print("Propagating Wavefunction")
        propagatorInstance.propagateCN(gridInstance,psiInstance,laserInstance,hamiltonianInstance)

        if comm.rank == 0:
            propend = time.time()
            print("Total Time to Propagate:",round(propend-propstart,4),"seconds")
        if comm.rank == 0 and DETAILS:
            print("##########################################################")
    
    if LOG_PROP:
        PETSc.Log.view()
    
    if comm.rank == 0:
        end = time.time()
        print("Total Simulation Time:",round((end-start)/60,4),"minutes")


        
        

        