import numpy as np
import load_data as ld
import math
import matplotlib.pyplot as plt
from MapUtils.MapUtils import getMapCellsFromRay
from deadReckoning import deadReckoning
from createMap import createMap

def SLAM(fileNumber):
    #number of particles
    n = 5

    #instantiate the particles
    positions = np.zeros((n,3))
    weights = (1.0/n)*np.ones((n,))

    #add the noise
    for i in range(0,n):
        #find some noise for each
        noise = np.random.normal(0,1e-4,(3,1))

        #do dead reckoning and get the poses
        deadRecokonedPoses = deadReckoning(fileNumber,noise)



        # plt.scatter(deadRecokonedPoses[:,0],deadRecokonedPoses[:,1],s=1)
        # plt.show()


if __name__ == "__main__":
    fileNumber = '0'
    SLAM(fileNumber)
