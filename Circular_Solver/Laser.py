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
    
    def createPulse(self):
        def A_funcX(t):
            A_x = self.E_0/self.w * self.env_func(t) * self.carrier_funcX(t) / np.sqrt(2)
            return A_x
        def A_funcY(t):
            A_y = self.E_0 /self.w* self.env_func(t) * self.carrier_funcY(t) / np.sqrt(2)
            return A_y
        self.A_funcX = A_funcX
        self.A_funcY = A_funcY

    def plotPulse(self,bool,gridInstance):
        if bool:
            t = gridInstance.t
            plt.figure()
            plt.plot(t,self.A_funcX(t),label = "X")
            plt.plot(t,self.A_funcY(t),label = "Y")
            plt.savefig("images/pulse.png")
            plt.clf()