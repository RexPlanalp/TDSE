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
    
    def createCarrierX(self,simInstance):
        w = simInstance.laser["w"]

        def carrierFuncX(t):
            return np.cos(w*t)
        self.carrier_funcX = carrierFuncX
        return
    
    def createCarrierY(self,simInstance):
        w = simInstance.laser["w"]
        def carrierFuncY(t):
            return np.sin(w*t)
        self.carrier_funcY = carrierFuncY
        return
    
    def createAmplitude(self,simInstance):
        I = simInstance.laser["I"]/3.51E16


        E_0 = np.sqrt(I)
        self.E_0 = E_0
        return 
    
    def createPulse(self,simInstance):
        w = simInstance.laser["w"]
        ell = simInstance.laser["ell"]
        def A_funcX(t):
            A_x = self.E_0/w * self.env_func(t) * self.carrier_funcX(t) / np.sqrt(1+np.abs(ell))
            return A_x
        def A_funcY(t):
            A_y = ell*self.E_0 /w* self.env_func(t) * self.carrier_funcY(t) / np.sqrt(1+np.abs(ell))
            return A_y
        self.A_funcX = A_funcX
        self.A_funcY = A_funcY

    def plotPulse(self,simInstance):
        Nt = simInstance.Nt
        time_size = simInstance.time_size
        t = np.linspace(0,time_size,Nt)

        plt.figure()
        plt.plot(t,self.A_funcX(t),label = "X")
        plt.plot(t,self.A_funcY(t),label = "Y")
        plt.savefig("images/pulse.png")
        plt.clf()
    
        
        plt.figure()
        plt.scatter(self.A_funcX(t),self.A_funcY(t),c=t,cmap="viridis",s = 0.5)
        plt.colorbar()
        plt.xlabel("X Component")
        plt.ylabel("Y Component")
        plt.gca().set_aspect('equal')
        plt.savefig("images/polar_pulse.png")
        plt.clf()

        np.save("TDSE_files/A_funcX.npy",self.A_funcX(t))
        np.save("TDSE_files/A_funcY.npy",self.A_funcY(t))
        np.save("TDSE_files/t.npy",t)

if __name__ == "__main__":
    laserInstance = laser()
    laserInstance.createEnvelope()
    laserInstance.createCarrierX()
    laserInstance.createCarrierY()
    laserInstance.createAmplitude()
    laserInstance.createPulse()
    laserInstance.plotPulse(True)