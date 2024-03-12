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
    
    def createCarrierX(self):
        def carrierFunc(t):
            return np.cos(self.w*t)
        self.carrier_funcX = carrierFunc
        return
    def createCarrierY(self):
        def carrierFunc(t):
            return np.sin(self.w*t)
        self.carrier_funcY = carrierFunc
        return
    
    def createAmplitude(self):
        E_0 = np.sqrt(self.I)
        self.E_0 = E_0
        return 
    
    def createPulse(self,gridInstance):
        t = gridInstance.t
        dt = gridInstance.dt

        E_x = self.E_0 * self.env_func(t) * self.carrier_funcX(t) / np.sqrt(2)
        E_y = self.E_0 * self.env_func(t) * self.carrier_funcY(t) / np.sqrt(2)

        if self.gauge == "length":
            self.pulse_array = E_x,E_y
        else:
            #A_x = []
            #A_y = []
            #for i,t_val in enumerate(t):
                #A_x_val = -np.trapz(E_x[:i],dx = dt)
                #A_y_val = -np.trapz(E_y[:i],dx = dt)
                #A_x.append(A_x_val)
                #A_y.append(A_y_val)
            #A_x = np.array(A_x)
            #A_y = np.array(A_y)
            A_x = self.E_0/self.w * self.env_func(t) * self.carrier_funcX(t) / np.sqrt(2)
            A_y = self.E_0 /self.w* self.env_func(t) * self.carrier_funcY(t) / np.sqrt(2)
            self.pulse_array = A_x,A_y

     
        

        
    



            

        
            


    

    def plotPulse(self,bool):
        if bool:
            plt.figure()
            plt.plot(self.pulse_array[0])
            plt.plot(self.pulse_array[1])
            plt.savefig("images/pulse.png")
            plt.clf()