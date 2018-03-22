import math
import numpy as np
import load_data as ld
import cv2
import matplotlib.pyplot as plt
import pickle
from SLAM import SLAM
#from vtk_visualizer import *
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D


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
    #get the rotation and translations between the IR camera and the RGB camera
    IRToRGBRotation = IRToRGBData['R']
    IRToRGBTranslation = IRToRGBData['T']

    #import the SLAM data
    #NOTE THAT THESE FILES ARE FROM THE LAST SLAM ROUTINE RUN!
    #IF YOU WANT TO RUN TEXTUREMAP ON A NEW DATASET, YOU HAVE TO RERUN SLAM by uncommenting the following command
    #SLAM(lidarFilename,jointFilename)
    #If you've already run SLAM with the dataset, then just continue
    with open('allPoses.pickle', 'rb') as handle:
        allPoses = pickle.load(handle)

    with open('finalMap.pickle', 'rb') as handle:
        finalMap = pickle.load(handle)

    #get the RGB data
    rgbData = ld.get_rgb(rgbFilename)
    depthData = ld.get_depth(depthFilename)

    #initialize the occupancy grid
    MAP = {}
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -20  # meters
    MAP['ymin'] = -20
    MAP['xmax'] = 20
    MAP['ymax'] = 20
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.double)  # DATA TYPE: char or int8

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

        #transform the points in the image to 3D points to project
        xRGB = XRGB[0,:]
        yRGB = XRGB[1,:]
        zRGB = XRGB[2,:]

        #find the colors associated with the 3D points
        #convert back to the uv points
        uRGB = np.round(fRGBy*xRGB/zRGB + rgbXPrincipalPoint).astype('uint8')
        vRGB = np.round(fRGBx*yRGB/zRGB + rgbYPrincipalPoint).astype('uint8')

        #get the colors for the 3D Points XRGB
        rgbColors = image[uRGB,vRGB,:]

        #convert these colors to the global frame
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

        #convert the 3D RGB points to the global frame
        dBody = .93 + .33
        rotGlobal = rotzHomo(thetaPose, xPose, yPose, dBody)
        totalRotation = np.dot(rotGlobal,totalRotHead)

        #rotate the RGB image and the depth image to the global frame
        FourDPoints = np.vstack((XRGB,np.ones(XRGB.shape[1],)))
        rotated3DPoints = np.dot(totalRotation,FourDPoints)

        #find the ground plane on the rotated data via RANSAC or via thresholding
        thresh = 1
        indicesToKeep = rotated3DPoints[2,:]<thresh
        pointsToPaint = rotated3DPoints[0:3,indicesToKeep].astype('uint8')
        rgbValuesOfPointsToKeep = rgbColors[indicesToKeep,:]

        #paint onto a plane
        # fig = plt.figure()
        # ax = fig.add_subplot(111)#, projection='3d')
        # ax.scatter(pointsToPaint[0,:],pointsToPaint[1,:],c=rgbValuesOfPointsToKeep/255.0)
        # plt.show()

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
