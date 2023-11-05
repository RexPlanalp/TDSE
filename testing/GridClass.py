from petsc4py import PETSc
from mpi4py import MPI

import json
import time
import numpy as np
class Grid:
    def __init__(self,grid_size,grid_spacing,time_size,time_spacing):
        self.comm = PETSc.COMM_WORLD
        self.rank = self.comm.Get_rank()

        Nr = int(grid_size / grid_spacing) + 1

        r = PETSc.Vec().createMPI(Nr, comm=PETSc.COMM_WORLD)
        r.setFromOptions()
        r.setUp()

        self.r = r
        self.Nr = Nr
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing

        Nt = int(time_size / time_spacing) + 1

        t = PETSc.Vec().createMPI(Nt, comm=PETSc.COMM_WORLD)
        t.setFromOptions()
        t.setUp()

        self.t = t
        self.Nt = Nt
        self.time_size = time_size
        self.time_spacing = time_spacing
    def InitializeGrid(self):
        if self.rank == 0:
             start = time.time()

        begin, end = self.r.getOwnershipRange()
        for i in range(begin, end):
            self.r.setValue(i, self.grid_spacing * i)
        self.r.assemblyBegin()
        self.r.assemblyEnd()

        begin,end = self.t.getOwnershipRange()
        for i in range(begin,end):
            self.t.setValue(i,-self.time_size/2 + i * self.time_spacing)
        self.t.assemblyBegin()
        self.t.assemblyEnd()
        
        if self.rank == 0:
             end = time.time()
             print(f"Time to create grid: {end-start} seconds")
    def PrintSaveGrid(self):
        if self.rank == 0:
            print(
                """
                Simulation Box:
            
                x = [{},{}], {}
                t = [0,{}], {}
            
                """.format(0,self.grid_size,self.grid_spacing,self.time_size,self.time_spacing)
            )

            r_array = self.r.getArray()
            t_array = self.t.getArray()

            np.save("r.npy",r_array)
            np.save("t.npy",t_array)
        return None
class Basis:
    def __init__(self,N_knots,order,N_gauss):
        self.N_knots = N_knots
        self.order = order
        self.N_gauss = N_gauss
    
    #def InitializeBasis(rmax):


     
     
          

        



if __name__ == "__main__":
    
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    

    box_par = tuple(input_par["box"].values())
    box = Grid(*box_par)
    box.InitializeGrid()
    #box.PrintSaveGrid()


    #splines_par = tuple(input_par["splines"].values())
    #splines = Basis(*splines_par)
    
    



