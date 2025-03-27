#from .pyCudaMetaLink import libpyCudaMeta as lpcm
import libpyCudaMeta as lpcm
#import calcKernel
#import enums

import numpy as np
from matplotlib import pyplot as plt
import pkgutil
import importlib
import os
import shutil
import glob
import sys
import configparser
import time
#from scipy.special import gamma as gammaFunction
#import scipy.fftpack as sp

class model(lpcm.Model):
    def __init__(self, 
                 stringSize = 1e6,
                 m1 = 0.02,
    ):
        lpcm.Model.__init__(self, stringSize)
    
    def setStringSize(self, stringSize):
        lpcm.Model.setStringSize(stringSize)

    def configure(self, configFile):
        config = configparser.ConfigParser()
        config.read(configFile)
        # Read values and convert to float
        parameters = np.zeros(18)
        parameters[0] = float(config["Parameters"]["k1"])
        parameters[1] = float(config["Parameters"]["k2"])
        parameters[2] = float(config["Parameters"]["k3"])
        parameters[3] = float(config["Parameters"]["m1"])
        parameters[4] = float(config["Parameters"]["m2"])
        parameters[5] = float(config["Parameters"]["m3"])
        parameters[6] = float(config["Parameters"]["L"])
        parameters[7] = float(config["Parameters"]["D"])
        parameters[8] = float(config["Parameters"]["g1"])
        parameters[9] = float(config["Parameters"]["g2"])
        parameters[10] = float(config["Parameters"]["g3"])
        parameters[11] = float(config["Parameters"]["R"])
        parameters[12] = float(config["Parameters"]["a"])
        parameters[13] = float(config["Parameters"]["alpha"])
        parameters[14] = float(config["Parameters"]["beta"])
        parameters[15] = float(config["Parameters"]["gamma"])
        parameters[16] = float(config["Parameters"]["r2"])
        parameters[17] = float(config["Parameters"]["dt"])
        self.setParameters(parameters)

    def setParameters(self, parameters):
        lpcm.Model.setParameters(self, parameters)

    def getParameters(self):
        return np.array(lpcm.Model.getParameters(self))

    def getStringSize(self):
        return lpcm.Model.getStringSize(self)

    def setPsi(self, psi):
        lpcm.Model.setPsi(self, psi)

    def getPsi(self):
        psi = np.array(lpcm.Model.getPsi(self))
        return psi

    def setPsiVel(self, psi):
        lpcm.Model.setPsiVel(self, psi)

    def getPsiVel(self):
        psiVel = np.array(lpcm.Model.getPsiVel(self))
        return psiVel

    def getPsiForces(self):
        psiForces = np.array(lpcm.Model.getPsiForces(self))
        return psiForces

    def getPsiLengths(self):
        psiLengths = np.array(lpcm.Model.getPsiLengths(self))
        return psiLengths

    def setPhi(self, psi):
        lpcm.Model.setPsi(self, psi)

    def getPhi(self):
        phi = np.array(lpcm.Model.getPhi(self))
        return phi

    def setPhiVel(self, phi):
        lpcm.Model.setPhiVel(self, phi)

    def getPhiVel(self):
        phiVel = np.array(lpcm.Model.getPhiVel(self))
        return phiVel

    def setTheta(self, theta):
        lpcm.Model.setPsi(self, psi)

    def getTheta(self):
        theta = np.array(lpcm.Model.getTheta(self))
        return theta

    def setThetaVel(self, theta):
        lpcm.Model.setThetaVel(self, theta)

    def getThetaVel(self):
        thetaVel = np.array(lpcm.Model.getThetaVel(self))
        return thetaVel

    def getV(self):
        v = np.array(lpcm.Model.getV(self))
        return v

    def getVVel(self):
        vVel = np.array(lpcm.Model.getVVel(self))
        return vVel

    def updatePsiForces(self, stepNum = -1):
        lpcm.Model.updatePsiForces(self, stepNum)

    def updatePsiPosVel(self, stepNum = -1):
        lpcm.Model.updatePsiPosVel(self, stepNum)

    def runSimulation(self, amplitude, omega, T):
        return lpcm.Model.runSimulation(self, amplitude, omega, T)

    def drive(self, amplitude, omega, t):
        lpcm.Model.drive(self, amplitude, omega, t)

    def setDeathCut(self, deathCut):
        lpcm.Model.setDeathCut(self, deathCut)

if __name__ == "__main__":
    stringSize = 1000
    m = model(stringSize = stringSize)
    m.configure("/home/rdennis/Documents/Code/pyCudaMetaAnalysis/testConfigs.ini")
    psi = np.zeros(stringSize)
    psiVel = np.zeros(stringSize)
    m.setPsi(psi)
    m.setPsiVel(psiVel)
    startTime = time.time()
    print("start ", time.time() - startTime)
    print(m.runSimulation(0.1, 0.05, 1000))
    print("finish ", time.time() - startTime)
    plt.plot(m.getPsi())
#    m.runSimulation(0.1, 0.01, 10)
    plt.show()
    print(m.getPsi()[-1])


