import numpy as np
import load_data as ld
import matplotlib.pyplot as plt


def deadReckoning(fileNumber):
    lidarFilename = 'data/train_lidar' + fileNumber
    lidarData = ld.get_lidar(lidarFilename)

    # get the total number of timestamps
    numTimeStamps = len(lidarData)

    #instantiate initial guess of X and of theta
    initPose = np.array(lidarData[0]['pose']).T
    xMinus = np.array([initPose[0][0], initPose[1][0], 0]).T
    thetaMinus = initPose[2][0]

    allPts = np.empty((numTimeStamps,3))

    for i in range(1, numTimeStamps-1):
        #load the data for this timestamp
        dataI = lidarData[i]

        #get the pose at time t
        pose = np.array(dataI['pose']).T
        xPose = pose[0][0]
        yPose = pose[1][0]
        thetaPose = pose[2][0]
        pt = np.array([xPose, yPose,0]).T

        #get the pose at time t+1
        nextPose = np.array(lidarData[i+1]['pose']).T
        nextXPose = nextPose[0][0]
        nextYPose = nextPose[1][0]
        nextThetaPose = nextPose[2][0]
        nextPt = np.array([nextXPose, nextYPose,0]).T

        #get the delta pose and delta thetas
        diffVectors = nextPt - pt
        rotMatrix = rotz(thetaPose)
        deltaX = np.dot(rotMatrix.T,diffVectors)
        deltaTheta = nextThetaPose - thetaPose

        #now go back to global
        xPlus = np.dot(rotz(thetaMinus),deltaX)+ xMinus
        thetaPlus = thetaMinus + deltaTheta
        xNew = xPlus[0]
        yNew = xPlus[1]

        #set thetaPlus equal to thetaMinus and
        thetaMinus = thetaPlus
        xMinus = xPlus

        allPts[i,0] = xNew
        allPts[i,1] = yNew
        allPts[i,2] = thetaMinus

    # plt.scatter(allPts[:,0],allPts[:,1],s=1)
    # plt.show()

    return allPts

def rotz(angle):
    return np.array([[np.cos(angle), - np.sin(angle),0],
            [np.sin(angle), np.cos(angle),0],
            [0, 0 , 1]])


if __name__ == "__main__":
    fileNumber = '0'
    deadReckoning(fileNumber)
