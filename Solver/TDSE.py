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
        basisInstance.createFuncs(gridInstance)
        basisInstance.plotFuncs(gridInstance,False)
        basisInstance.createGauss(gridInstance)
        basisInstance.evalGauss()

        if comm.rank == 0:
            basisend = time.time()
            print("Total Time to Construct Basis:",basisend-basisstart)


    if TISE:
        if comm.rank == 0:
            tisestart = time.time()
    
        tiseInstance = tise()
        tiseInstance.createS_R(basisInstance)
        tiseInstance.createAllH(basisInstance)
        tiseInstance.solveEigensystem()

        if comm.rank == 0:
            tiseend = time.time()
            print("Total Time to Find Eigensystem:",tiseend-tisestart)


    if LASER:
        if comm.rank == 0:
            laserstart = time.time()
    
        laserInstance = laser()
        laserInstance.createPulse(gridInstance)
        laserInstance.plotPulse(False)

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
        hamiltonianInstance.PartialAngular()
        
        if comm.rank == 0:
            hamend = time.time()
            print("Total Time to Create Interaction:",hamend-hamstart)


    if PROPAGATE:

        if comm.rank == 0:
            propstart = time.time()

        propagatorInstance = propagator()
        propagatorInstance.propagateCN(gridInstance,psiInstance,laserInstance,hamiltonianInstance)

        if comm.rank == 0:
            propend = time.time()
            print("Total Time to Propagate:",propend-propstart)
    
    

    if comm.rank == 0:
        end = time.time()
        print("Total Simulation Time:",end-start)

    
    TESTONE = False # Testing norm of unembedded initial state vs embedded
    if TESTONE:

        # First we retrieve the unembedded state from the TISE output
        with h5py.File('Hydrogen.h5', 'r') as f:
            data = f[f"/Psi_{1}_{0}"][:]
            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part


        inner_prod = 0
        seq_S = getLocal(tiseInstance.S_R)
        for i,ci in enumerate(total):
            for j,cj in enumerate(total):
                inner_prod += np.conjugate(cj)*ci * seq_S.getValue(i,j)
        if PETSc.COMM_WORLD.rank == 0:
            print(inner_prod)

        psi_initial = PETSc.Vec().createWithArray(comm = PETSc.COMM_WORLD,size = len(total),array = total)

        Sv = tiseInstance.S_R.getVecRight()
        tiseInstance.S_R.mult(psi_initial,Sv)
        inner_prod2 = psi_initial.dot(Sv)
        if comm.rank == 0:
            print(inner_prod2)

    TESTTWO = False # Testing norm of embedded initial state
    if TESTTWO:
        Sv = hamiltonianInstance.S.getVecRight()
        hamiltonianInstance.S.mult(psiInstance.psi_initial,Sv)
        
        inner_prod = psiInstance.psi_initial.dot(Sv)
        if comm.rank == 0:
            print(inner_prod)

    TESTTHREE = False # Testing norm of embedded final state
    if TESTTHREE:
        Sv = hamiltonianInstance.S.getVecRight()
        hamiltonianInstance.S.mult(psiInstance.psi_final,Sv)

        inner_prod = psiInstance.psi_initial.dot(Sv)

        if comm.rank == 0:
            print(np.abs(inner_prod)**2)


    TESTFOUR = True
    if TESTFOUR:
        mat = PETSc.Mat()
        viewer = PETSc.Viewer().createBinary('overlap.bin', 'r')
        mat.load(viewer)
        viewer.destroy()