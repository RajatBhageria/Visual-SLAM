import math
import numpy as np
import load_data as ld
import cv2
import matplotlib.pyplot as plt
import pickle

def textureMap(lidarFilepath,jointFilepath,rgbFilename,depthFilename):
    #get the calibration data
    rgbCalibFile = "train_rgb/cameraParam/IRcam_Calib_result.pkl"
    depthCalibFile= "train_rgb/cameraParam/RGBcamera_Calib_result.pkl"
    IRToRGBFile = "train_rgb/cameraParam/exParams.pkl"
    with open(rgbCalibFile, 'rb') as handle:
        rgbCalib = pickle.load(handle)

    with open(depthCalibFile, 'rb') as handle:
        depthCalib = pickle.load(handle)

    #get the rgb calibration data
    fRGB = rgbCalib['fc']
    fRGBy = fRGB[0]
    fRGBx = fRGB[1]
    rgbPrincipalPoint = rgbCalib['cc']
    rgbYPrincipalPoint = rgbPrincipalPoint[0]
    rgbXPrincipalPoint = rgbPrincipalPoint[1]

    #get the IR calibratin data
    fIR = depthCalib['fc']
    fIRy = fIR[0]
    fIRx = fIR[1]
    depthPrincipalPoint = rgbCalib['cc']
    depthYPrincipalPoint = depthPrincipalPoint[0]
    depthXPrincipalPoint = depthPrincipalPoint[1]

    #get the transformation from IR to RGB
    with open(IRToRGBFile, 'rb') as handle:
        IRToRGBData = pickle.load(handle)

    IRToRGBRotation = IRToRGBData['R']
    IRToRGBTranslation = IRToRGBData['T']

    #import the SLAM data
    with open('allPoses.pickle', 'rb') as handle:
        allPoses = pickle.load(handle)

    with open('finalMap.pickle', 'rb') as handle:
        finalMap = pickle.load(handle)

    #get the RGB data
    rgbData = ld.get_rgb(rgbFilename)
    depthData = ld.get_depth(depthFilename)

    #number of images
    numImages = len(rgbData)
    for i in range(0,numImages):
        image = rgbData[i]['image'] #1920 x 1080 x 3
        depth = depthData[i]['depth']# 412 x 512
        time = rgbData[i]['t']

        #create a meshgrid of the depth to do the transformations
        row = np.arange(0, depth.shape[0],1)
        col = np.arange(0, depth.shape[1],1)
        cols, rows = np.meshgrid(col, row)

        #do the RGB and depth image alignment
        #convert the IR data to 3Dpoints
        xIR = np.multiply(cols-depthXPrincipalPoint,depth)/fIRx
        yIR = np.multiply(rows-depthYPrincipalPoint,depth)/fIRy

        #convert image into X vector of all xyz locations
        X = np.vstack((np.array(xIR).ravel(),np.array(yIR).ravel(),np.array(depth).ravel()))

        #transform the IR data into the RGB image frame using the camera parameters
        XRGB = np.dot(IRToRGBRotation,X) + IRToRGBTranslation

        #transform the points in the image to 3D points
        xRGB = XRGB[0,:]
        yRGB = XRGB[1,:]
        zRGB = XRGB[2,:]g

        uRGB = fRGBy*xRGB/zRGB + rgbXPrincipalPoint
        vRGB = fRGBx*yRGB/zRGB + rgbYPrincipalPoint

        #find the rotations for head and yaw angles
        #find the head angles of the closest times
        neckAngle = rgbData[i]['head_angles'].T[0]
        headAngle = rgbData[i]['head_angles'].T[1]

        #find the position of the body at this time
        dNeck = .07
        rotNeck = rotzHomo(neckAngle,0,0,dNeck)
        rotHead = rotyHomo(headAngle)
        totalRotHead = np.dot(rotNeck,rotHead)

        #find the closest timestep from the SLAM data
        idx = findIndexOfCloestTimeFrame(allPoses[:,0],time)

        #find the closest pose x,y position from the SLAM data
        pose = allPoses[idx,:]
        xPose = pose[1]
        yPose = pose[2]
        thetaPose = pose[3]

        #convert to the global frame
        dBody = .93 + .33
        rotGlobal = rotzHomo(thetaPose, xPose, yPose, dBody)
        totalRotation = np.dot(rotGlobal,totalRotHead)

        #rotate the RGB image and the depth image to the global frame
        rotatedImage = cv2.warpPerspective(image,totalRotMatrix,image.shape[0:2])
        plt.imshow(rotatedImage)
        plt.show()


        #find the ground plane on the rotated data via RANSAC or via thresholding


        #


def findIndexOfCloestTimeFrame(jointTimes, ts):
    idx = (np.abs(jointTimes - ts)).argmin()
    return idx

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


if __name__ == "__main__":
    fileNumber = '0'
    lidarFilename = 'data/train_lidar'+fileNumber
    jointFilename = 'data/train_joint'+fileNumber
    rgbFilename = 'train_rgb/RGB_' +fileNumber
    depthFilename = 'train_rgb/DEPTH_' +fileNumber
    textureMap(lidarFilename,jointFilename,rgbFilename,depthFilename)
