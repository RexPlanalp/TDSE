import matplotlib.pyplot as plt
import numpy as np


class laser:
    def __init__(self):
        return None
    
    def createEnvelope(self,simInstance):
        envelope = simInstance.laser["envelope"]
        w = simInstance.laser["w"]

        N = simInstance.box["N"]

        if envelope == "sinsq":
            def envFunc(t):
                env = np.sin(w * t/(2*N))**2
                return env
        self.env_func = envFunc
    
    def createCarrier(self,simInstance):
        w = simInstance.laser["w"]

        def carrierFunc(t):
            return np.cos(w*t)
        self.carrier_func = carrierFunc
        
    def createAmplitude(self,simInstance):
        I = simInstance.laser["I"]/3.51E16

        E_0 = np.sqrt(I)
        self.E_0 = E_0
        return 
    
    def createPulse(self,simInstance):
        w = simInstance.laser["w"]
        def A_func(t):
            A = self.E_0/w * self.env_func(t) * self.carrier_func(t)
            return A
        self.A_func = A_func
        return None
        
    def plotPulse(self,simInstance):
        Nt = simInstance.Nt
        time_size = simInstance.time_size
        t = np.linspace(0,time_size,Nt)

        # plt.figure()
        # plt.plot(t,self.A_func(t),label = "Z")
        # plt.savefig("images/pulse.png")
        # plt.clf()

        # np.save("TDSE_files/A_func.npy",self.A_func(t))
        # np.save("TDSE_files/t.npy",t)

if __name__ == "__main__":
    pass