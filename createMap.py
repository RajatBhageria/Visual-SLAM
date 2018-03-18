import numpy as np
import load_data as ld
import math
import matplotlib.pyplot as plt
from MapUtils.MapUtils import getMapCellsFromRay

def createMap(fileNumbe):
    lidarFilename = 'data/train_lidar'+fileNumber
    jointFileName = 'data/train_joint'+fileNumber
    lidarData = ld.get_lidar(lidarFilename)
    jointData = ld.get_joint(jointFileName)

    #find all the joint times
    allJointTimes = jointData['ts'][0]

    #find all the head angles from joint data
    headAngles = jointData['head_angles']

    #get the total number of timestamps
    numTimeStamps = len(lidarData)

    # init MAP
    MAP = {}
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -20  # meters
    MAP['ymin'] = -20
    MAP['xmax'] = 20
    MAP['ymax'] = 20
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8

    for i in range(0,numTimeStamps):
        #load the data for this timestamp
        dataI = lidarData[i]

        #find the thetas
        theta = np.array([np.arange(-135,135.25,0.25)*np.pi/180.])

        #get the scan at this point
        di = np.array(dataI['scan'])

        # take valid indices
        # indValid = np.logical_and((di < 30), (di > 0.1))
        # print indValid.shape
        # theta = theta[indValid]
        # di = di[indValid]

        #find the position of the head at this time
        rHead = np.vstack([di * np.sin(theta), di * np.cos(theta), np.zeros((1,1081))])

        #find the time stamp
        ts = dataI['t'][0][0]

        #find the closest joint based on the absolute time
        idx = findIndexOfCloestTimeFrame(allJointTimes,ts)

        #find the head angles of the closest times
        neckAngle = headAngles[0][idx]
        headAngle = headAngles[1][idx]

        #find the position of the body at this time
        dNeck = .15 + .33
        dBody = .93
        rotations = rotz(neckAngle)*roty(headAngle)
        rBody = np.dot(rotations,(rHead+dNeck)) + dBody

        #remove the data of the lidar hitting the ground
        rBody = rBody[:,rBody[2,:] >= 1E-4]

        #convert from body frame to the global frame
        pose = np.array(dataI['pose']).T
        xPose = pose[0][0]
        yPose = pose[1][0]

        #find the yaw and pitch angle of the lidar
        rpy = np.array(dataI['rpy']).T
        yawAngle = rpy[2]
        #find the rotation matrix for this
        rotGlobal = rotzHomo(yawAngle,xPose,yPose)
        threeDPoints = np.vstack((rBody,np.zeros((1,rBody.shape[1]))))
        rGlobal = np.dot(rotGlobal,threeDPoints)
        rGlobal = rGlobal[0:2,:]

        #convert to cells
        xs0 = rGlobal[0,:]
        ys0 = rGlobal[1,:]
        xis = np.ceil((xs0 - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        yis = np.ceil((ys0 - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        #run getMapCellsFromRay to get points occupied and unoccipied
        # cells = getMapCellsFromRay(xPose,yPose,xis,yis)
        # occupiedCells = cells[0,:]
        # freeCells = cells[1,:]

        #add the data points to the map
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
        MAP['map'][xis[indGood], yis[indGood]] = 1



    fig2 = plt.figure(2);
    plt.imshow(MAP['map'], cmap="hot");
    plt.show()

def rotz(angle):
    return np.vstack([[math.cos(angle), - math.sin(angle),0],
            [math.sin(angle), math.cos(angle),0],
            [0, 0 , 1]])

def rotzHomo(angle,tx,ty):
    return np.vstack([[math.cos(angle), - math.sin(angle),0,tx],
                      [math.sin(angle), math.cos(angle),0,ty],
                      [0, 0 , 1, 0],
                      [0, 0, 0, 1]])

def roty(angle):
    return np.vstack([[math.cos(angle),0, math.sin(angle)],
            [0,            1,      0],
            [-math.sin(angle),0,math.cos(angle)]])


def findIndexOfCloestTimeFrame(jointTimes, ts):
    minDist = 1000000000
    idx = (np.abs(jointTimes - ts)).argmin()
    return idx

if __name__ == "__main__":
    fileNumber = '0'
    createMap(fileNumber)
