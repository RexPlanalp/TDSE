import numpy as np
from numpy import pi
import json

class grid:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)

        grid_size = input_par["box"]["xmax"]
        grid_spacing = input_par["box"]["dx"]
        freq = input_par["laser"]["w"]
        N = input_par["box"]["N"]
        time_spacing = input_par["box"]["dt"]

        self.r = np.linspace(0,grid_size,int(grid_size/grid_spacing)) 

        ###########
        self.r = np.linspace(0,grid_size,int(grid_size/grid_spacing)+1) 
        ###########

        self.tau = 2*pi/freq
        self.tmax = N*self.tau
        self.t = np.arange(-self.tmax/2,self.tmax/2 + time_spacing,time_spacing)

        ##############
        N = int(np.rint(self.tmax / time_spacing)) + 1
        self.t = np.linspace(-self.tmax/2,self.tmax/2,N)
        ##############


        self.rmax = grid_size
        self.dr = grid_spacing
        self.dt = time_spacing
    def print(self,bool):
        if bool:
            print(
                """
                Simulation Box:
            
                x = [{},{}], {}
                t = [{},{}], {}
            
                """.format(0,self.rmax,self.dr,np.min(self.t),np.max(self.t),self.dt)
            )
        return None  
    

if __name__ == "__main__":
    gridInstance = grid()
    gridInstance.print(True)