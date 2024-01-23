from numpy import pi
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.integrate import simps

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
    
   
    

    def createEnvelope(self):
        if self.envelope == "sinsq":
            def envFunc(t):
                env = np.sin(self.w * t/(2*self.N))**2
                return env
        self.env_func = envFunc
    
    def createCarrier(self):
        def carrierFunc(t):
            return np.cos(self.w*t)
        self.carrier_func = carrierFunc
        return
    
    def createAmplitude(self):
        E_0 = np.sqrt(self.I)
        self.E_0 = E_0
        return 
    
    def createPulse(self,gridInstance):
        t = gridInstance.t
        dt = gridInstance.dt

        E = self.E_0 * self.env_func(t) * self.carrier_func(t)

        if self.gauge == "length":
            self.pulse_array = E
        else:
            A = []
            for i,t_val in enumerate(t):
                A_val = -np.trapz(E[:i],dx = dt)
                A.append(A_val)
            A = np.array(A)
            self.pulse_array = A

     
        

        
    



            

        
            


    

    def plotPulse(self,bool):
        if bool:
            plt.figure()
            plt.plot(self.pulse_array)
            plt.savefig("images/pulse.png")
            plt.clf()