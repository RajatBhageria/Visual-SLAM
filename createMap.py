import numpy as np
import load_data as ld
import math
import matplotlib.pyplot as plt
from MapUtils.MapUtils import getMapCellsFromRay
from deadReckoningOnly import deadReckoning

def createMap(fileNumber):
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
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.double)  # DATA TYPE: char or int8

    deadReckoningPoses = deadReckoning(fileNumber)

    for i in range(0,numTimeStamps,100):
        #load the data for this timestamp
        dataI = lidarData[i]

        #find the thetas
        theta = np.array([np.arange(-135,135.25,0.25)*np.pi/180.])

        #get the scan at this point
        di = np.array(dataI['scan'])

        # take valid indices
        # indValid = np.logical_and((di < 30), (di > 0.1))
        # theta = theta[indValid]
        # di = di[indValid]

        #find the position of the head at this time
        rHead = np.vstack([di * np.cos(theta), di * np.sin(theta), np.zeros((1,1081)),np.ones((1,1081))])

        #find the time stamp
        ts = dataI['t'][0][0]

        #find the closest joint based on the absolute time
        idx = findIndexOfCloestTimeFrame(allJointTimes,ts)

        #find the head angles of the closest times
        neckAngle = headAngles[0][idx]
        headAngle = headAngles[1][idx]

        #find the position of the body at this time
        dNeck = .15
        rotNeck = rotzHomo(neckAngle,0,0,dNeck)
        rotHead = rotyHomo(headAngle)
        rBody = np.dot(np.dot(rotNeck,rotHead),rHead)

        #get the body roll and pitch angles
        rollBody = jointData['rpy'][0][i]
        pitchBody = jointData['rpy'][1][i]

        rotBodyRoll = rotxHomo(rollBody,0,0,0)
        rotBodyPitch = rotyHomo(pitchBody)

        #apply the body roll and pitch angles
        rBody = np.dot(rotBodyPitch,np.dot(rotBodyRoll,rBody))

        #get the dead recokoned poses
        xPose = deadReckoningPoses[i][0]
        yPose = deadReckoningPoses[i][1]
        thetaPose = deadReckoningPoses[i][2]

        #find the yaw and pitch angle of the lidar
        rpy = np.array(dataI['rpy']).T
        yawAngle = rpy[2]

        #convert from the body frame to the global frame
        dBody = .93 + .33
        rotGlobal = rotzHomo(yawAngle,xPose,yPose,dBody)
        rGlobal = np.dot(rotGlobal,rBody)

        #remove the data of the lidar hitting the ground
        rGlobal = rGlobal[:,(rGlobal[2,:] >= 1E-4)]

        #convert to cells
        xs0 = rGlobal[0,:]
        ys0 = rGlobal[1,:]
        xis = np.ceil((xs0 - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        yis = np.ceil((ys0 - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        xPose = np.ceil((xPose - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        yPose = np.ceil((yPose - MAP['xmin']) / MAP['res']).astype(np.int16) - 1

        #run getMapCellsFromRay to get points that are unoccupied
        cells = getMapCellsFromRay(xPose,yPose,xis,yis)
        xsFree = np.int_(cells[0,:])
        ysFree = np.int_(cells[1,:])

        #find all the occupied cells
        xsOcc = xis
        ysOcc = yis

        #increase log odds of unoccupied cells to the map with log odds
        indGood = np.logical_and(np.logical_and(np.logical_and((xsFree > 1), (ysFree > 1)), (xsFree < MAP['sizex'])), (ysFree < MAP['sizey']))
        logOddsStepDecease = .008
        MAP['map'][xsFree[indGood], ysFree[indGood]] = MAP['map'][xsFree[indGood], ysFree[indGood]] - logOddsStepDecease

        #decrease log odds of occupied cells
        indGoodOcc = np.logical_and(np.logical_and(np.logical_and((xsOcc > 1), (ysOcc > 1)), (xsOcc < MAP['sizex'])), (ysOcc < MAP['sizey']))
        logOddsStepIncrease = .05
        MAP['map'][xsOcc[indGoodOcc], ysOcc[indGoodOcc]] = MAP['map'][xsOcc[indGoodOcc], ysOcc[indGoodOcc]] + logOddsStepIncrease

    #show the map
    plt.imshow(MAP['map'], cmap="hot");
    posesX = np.ceil((deadReckoningPoses[:,0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    posesY = np.ceil((deadReckoningPoses[:,1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    #show the dead reckoned poses
    plt.scatter(posesX,posesY)
    plt.show()

# def rotz(angle):
#     return np.vstack([[math.cos(angle), - math.sin(angle),0],
#             [math.sin(angle), math.cos(angle),0],
#             [0, 0 , 1]])
# def roty(angle):
#     return np.vstack([[math.cos(angle),0, math.sin(angle)],
#             [0,            1,      0],
#             [-math.sin(angle),0,math.cos(angle)]])

def rotzHomo(angle,tx,ty,tz):
    return np.vstack([[math.cos(angle), - math.sin(angle),0,tx],
                      [math.sin(angle), math.cos(angle),0,ty],
                      [0, 0 , 1, tz],
                      [0, 0, 0, 1]])

def rotyHomo(angle):
    return np.vstack([[math.cos(angle),0, math.sin(angle),0],
            [0,            1,      0,0],
            [-math.sin(angle),0,math.cos(angle),0],
            [0, 0, 0, 1]])


def rotxHomo(angle,tx,ty,tz):
    return np.vstack([[1,0,0,tx],
                      [0,math.cos(angle), - math.sin(angle),ty],
                      [0,math.sin(angle), math.cos(angle), tz],
                      [0, 0, 0, 1]])

def findIndexOfCloestTimeFrame(jointTimes, ts):
    idx = (np.abs(jointTimes - ts)).argmin()
    return idx

if __name__ == "__main__":
    fileNumber = '1'
    createMap(fileNumber)
