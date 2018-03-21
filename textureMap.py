
import numpy as np
import load_data as ld
import math
import matplotlib.pyplot as plt
from MapUtils.MapUtils import getMapCellsFromRay
from deadReckoning import deadReckoning
from MapUtils.MapUtils import mapCorrelation

def textureMap(lidarFilepath,jointFilepath,rgbFilename,depthFilename):
    #get the data
    lidarData = ld.get_lidar(lidarFilepath)
    jointData = ld.get_joint(jointFilepath)

    #find all the joint times
    allJointTimes = jointData['ts'][0]

    #find all the head angles from joint data
    headAngles = jointData['head_angles']

    #get the total number of timestamps
    numTimeStamps = len(lidarData)

    #get the RGB data
    rgbData = ld.get_rgb(rgbFilename)
    depthData = ld.get_depth(depthFilename)

    #number of images
    numImages = len(rgbData)
    for i in range(0,numImages):
        image = rgbData[i]['image'] #1920 x 1080 x 3
        depth = depthData[i]['depth'] #412 x 512
        time = rgbData[i]['t']

        #find the closest slam data


def findIndexOfCloestTimeFrame(jointTimes, ts):
    idx = (np.abs(jointTimes - ts)).argmin()
    return idx


if __name__ == "__main__":
    fileNumber = '0'
    lidarFilename = 'data/train_lidar'+fileNumber
    jointFilename = 'data/train_joint'+fileNumber
    rgbFilename = 'train_rgb/RGB_' +fileNumber
    depthFilename = 'train_rgb/DEPTH_' +fileNumber
    textureMap(lidarFilename,jointFilename,rgbFilename,depthFilename)
