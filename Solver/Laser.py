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
                env = np.sin(self.w * (t-tmax/2)/(2*self.N))**2
                return env
        self.env_func = envFunc
        
    def prepareFreq(self,gridInstance):

        t = gridInstance.t

    
        t_0_numerator = trapz(t*self.env_func(t),t)
        t_0_denominator = trapz(self.env_func(t),t)

        t_0 = t_0_numerator/t_0_denominator

        G_numerator = trapz((t-t_0)**2 * self.env_func(t),t)
        G_denominator = trapz(self.env_func(t),t)

        G = G_numerator/G_denominator

        F = self.w * np.sqrt(G)

        H = (1+np.sqrt(1+4/F**2))/2

        self.w = self.w/H


            

        
            


    def createPulse(self,gridInstance):
        t = gridInstance.t
        tmax = gridInstance.tmax
        if self.envelope == "sinsq":
            def pulseFunc(t_i):
                env = np.sin(self.w * (t_i-tmax/2)/(2*self.N))**2
                amplitude = pow(self.I, 0.5) / self.w
                if self.gauge == "length":
                    amplitude *= self.w
                else:
                    pass
                weighted_env = amplitude * env
                pulse = weighted_env*np.sin(self.w * (t_i-tmax/2))
                return pulse
            self.pulse_func = pulseFunc
        self.pulse = self.pulse_func(t)
        
        return 

    def plotPulse(self,bool):
        if bool:
            plt.figure()
            plt.plot(self.pulse)
            plt.savefig("images/pulse.png")
            plt.clf()