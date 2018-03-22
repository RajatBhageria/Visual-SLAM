import numpy as np
import load_data as ld
import math
import matplotlib.pyplot as plt
from MapUtils.MapUtils import getMapCellsFromRay
from deadReckoning import deadReckoning
from MapUtils.MapUtils import mapCorrelation
import pickle

def finalSLAM(lidarFilepath,jointFilepath):
    #get the data
    lidarData = ld.get_lidar(lidarFilepath)
    jointData = ld.get_joint(jointFilepath)

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

    # x-positions of each pixel of the map
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])
    # y-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])

    # find the thetas
    theta = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.])

    # number of particles
    numParticles = 10

    # instantiate the particles
    particlePoses = np.zeros((numParticles, 3))
    weights = (1.0 / numParticles) * np.ones((numParticles,))

    fig = plt.figure()
    im = plt.imshow(MAP['map'], cmap="hot", animated=True)

    #all predicted robot positions
    finalPoses = np.zeros((numTimeStamps,4))

    for i in range(0,numTimeStamps-1):
        #load the data for this timestamp
        dataI = lidarData[i]

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

        #get the pose of the best particle p*
        indexOfBest = np.argmax(weights)
        position = particlePoses[indexOfBest,:]

        #get the position of the best particles
        xPose = position[0]
        yPose = position[1]
        thetaPose = position[2]

        #add the current prediction of x y to numTimeStepsx2 matrix to plot later
        finalPoses[i][0] = ts
        finalPoses[i][1] = xPose
        finalPoses[i][2] = yPose
        finalPoses[i][3] = thetaPose

        #find the yaw and pitch angle of IMU rpy
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
        logOddsStepDecease = 1#np.log(.8/.3)
        MAP['map'][xsFree[indGood], ysFree[indGood]] = MAP['map'][xsFree[indGood], ysFree[indGood]] - logOddsStepDecease

        #decrease log odds of occupied cells
        indGoodOcc = np.logical_and(np.logical_and(np.logical_and((xsOcc > 1), (ysOcc > 1)), (xsOcc < MAP['sizex'])), (ysOcc < MAP['sizey']))
        logOddsStepIncrease = 3#np.log(.7/.2)#.05
        MAP['map'][xsOcc[indGoodOcc], ysOcc[indGoodOcc]] = MAP['map'][xsOcc[indGoodOcc], ysOcc[indGoodOcc]] + logOddsStepIncrease

        #do localization prediction
        # get the odometry for the i data
        odomCurr = np.array(dataI['pose']).T
        xOdom = odomCurr[0][0]
        yOdom = odomCurr[1][0]
        thetaOdom = odomCurr[2][0]
        currOdomPose = np.array([xOdom, yOdom]).T

        # get the odometry for the i+1 data
        odomNext = np.array(lidarData[i+1]['pose']).T
        nextXOdom = odomNext[0][0]
        nextYOdom = odomNext[1][0]
        nextThetaOdom = odomNext[2][0]
        nextOdomPose = np.array([nextXOdom, nextYOdom]).T

        #do prediction step for particles to get particle posotions at t+1
        particlePoses = deadReckoning(particlePoses,currOdomPose,thetaOdom,nextOdomPose,nextThetaOdom)

        #do localization update and get updated weights for particles
        for particleI in range(0,numParticles):
            #get particleI at time t+1
            particle = particlePoses[particleI,:]
            #get x and y positions of particle at time t+1
            particleX = particle[0]
            particleY = particle[1]

            #create a 9x9 window around the particle position
            x_range = np.arange(particleX-0.2, particleX + 0.2, 0.05)
            y_range = np.arange(particleY-0.2, particleY + 0.2, 0.05)

            #get the correlation between window and the map
            corr = mapCorrelation(MAP['map'], x_im, y_im, rGlobal[0:3,:], x_range,y_range)
            maxCorr = np.max(corr[:])
            oldWeight = weights[particleI]

            #get the new weight
            newWeight = oldWeight * maxCorr

            #update new weight in weights
            weights[particleI] = newWeight

        #normalize the weights
        weights = weights / np.sum(weights)

        #resample the particles
        Neff = 1.0/np.sum(weights**2)
        Nthresh = 20
        #do uniform sampling
        if Neff < Nthresh:
            weights = (1.0 / numParticles) * np.ones((numParticles,))

        #if you want to visualize progress happening
        #print i

    #show the map
    plt.imshow(MAP['map'], cmap="hot")

    #show the localized path
    posesX = np.ceil((finalPoses[:,1] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    posesY = np.ceil((finalPoses[:,2] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    plt.scatter(posesX,posesY,s=1)

    #show the plot
    plt.show()

    #save the poses and map as a pickle file
    with open('allPoses.pickle', 'wb') as handle:
        pickle.dump(finalPoses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('finalMap.pickle', 'wb') as handle:
        pickle.dump(MAP, handle, protocol=pickle.HIGHEST_PROTOCOL)

def rot(angle):
    return np.vstack([[math.cos(angle), - math.sin(angle)],
            [math.sin(angle), math.cos(angle)]])


def rotzHomo(angle,tx,ty,tz):
    return np.vstack([[math.cos(angle), - math.sin(angle),0,tx],
                      [math.sin(angle), math.cos(angle),0,ty],
                      [0, 0 , 1, tz],
                      [0, 0, 0, 1]])


def rotxHomo(angle,tx,ty,tz):
    return np.vstack([[1,0,0,tx],
                      [0,math.cos(angle), - math.sin(angle),ty],
                      [0,math.sin(angle), math.cos(angle), tz],
                      [0, 0, 0, 1]])


def rotyHomo(angle):
    return np.vstack([[math.cos(angle),0, math.sin(angle),0],
            [0,            1,      0,0],
            [-math.sin(angle),0,math.cos(angle),0],
            [0, 0, 0, 1]])


def findIndexOfCloestTimeFrame(jointTimes, ts):
    idx = (np.abs(jointTimes - ts)).argmin()
    return idx

def deadReckoning(allParticles,currOdomPose,thetaOdom,nextOdomPose,nextThetaOdom):
    #do smart subtract
    # get the delta pose and delta thetas
    diffVectors = nextOdomPose - currOdomPose
    rotMatrix = rot(thetaOdom)
    deltaX = np.dot(rotMatrix.T, diffVectors)
    deltaTheta = nextThetaOdom - thetaOdom

    #find number of particles
    [numParticles,_] = allParticles.shape

    #create return type
    predictionForParticles = np.zeros(allParticles.shape)

    for i in range(0,numParticles):
        particleIPos = allParticles[i,:]
        thetaMinus = particleIPos[2]
        xMinus = particleIPos[0:2]

        #do smart add and add to initial particle position
        xPlus = np.dot(rot(thetaOdom), deltaX) + xMinus
        thetaPlus = thetaMinus + deltaTheta

        # add the noise
        noise = np.random.normal(0, 1e-2, (3, 1))
        thetaPlus = thetaPlus + noise[2, 0]
        xPlus = xPlus + noise[0:2,0]#np.hstack((noise[0:2, 0], 0))

        #set equal to the original matrix
        predictionForParticles[i,:] = np.hstack((xPlus[0:2], thetaPlus))

    return predictionForParticles

if __name__ == "__main__":
    fileNumber = '0'
    lidarFilename = 'data/train_lidar'+fileNumber
    jointFileName = 'data/train_joint'+fileNumber
    finalSLAM(lidarFilename,jointFileName)