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

        return None
    
    def createPulse(self,gridInstance):
        t = gridInstance.t
        tmax = gridInstance.tmax
        
        if self.envelope == "sinsq":
            self.env = np.power(np.sin(pi*(t-tmax/2) / tmax), 2.0)

        amplitude = pow(self.I, 0.5) / self.w
        weighted_env = amplitude * self.env
        pulse = weighted_env*np.sin(self.w*t)

        

        self.pulse = pulse

        return None
    def createPulse(self,gridInstance):
        t = gridInstance.t
        tmax = gridInstance.tmax
        if self.envelope == "sinsq":
            def pulseFunc(t_i):
                env = np.power(np.sin(pi*(t_i-tmax/2) / tmax), 2.0)
                amplitude = pow(self.I, 0.5) / self.w
                weighted_env = amplitude * env
                pulse = weighted_env*np.sin(self.w*t_i)
                return pulse
            self.pulse_func = pulseFunc
        self.pulse = self.pulse_func(t)
        
        return 


    def plotPulse(self,bool):
        if bool:
            plt.figure()
            plt.plot(self.pulse)
            plt.savefig("pulse.png")
            plt.clf()