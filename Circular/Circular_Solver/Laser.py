from numpy import pi
import matplotlib.pyplot as plt
import numpy as np
import json

class laser:


    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)

        self.I = input_par["laser"]["I"] / 3.51E16
        self.w = input_par["laser"]["w"]
        self.envelope = input_par["laser"]["envelope"]
        self.N = input_par["box"]["N"]
        self.gauge = input_par["laser"]["gauge"]

        return None
    
   
    


    def createPulse(self,gridInstance):
        t = gridInstance.t
        tmax = gridInstance.tmax
        if self.envelope == "sinsq":
            def pulseFunc(t_i):
                env = np.sin(self.w * (t_i-tmax/2)/(2*self.N))**2
                amplitude = pow(self.I, 0.5) / self.w
                amplitude *= 1/np.sqrt(2)
                weighted_env = amplitude * env

                pulse_x = weighted_env*np.sin(self.w * (t_i-tmax/2))
                pulse_y = weighted_env*np.cos(self.w * (t_i-tmax/2))
                return pulse_x,pulse_y
            self.pulse_func = pulseFunc
        self.pulse = self.pulse_func(t)
        return 
    
    


    def plotPulse(self,bool):
        if bool:
            plt.figure()
            plt.plot(self.pulse)
            plt.savefig("images/pulse.png")
            plt.clf()