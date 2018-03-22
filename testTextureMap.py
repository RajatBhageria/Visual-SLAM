from textureMap import textureMap

#note that you have to run SLAM.py on the lidar and joint data before calling tetureMap.

fileNumber = '0'
lidarFilename = 'data/train_lidar' + fileNumber
jointFilename = 'data/train_joint' + fileNumber
rgbFilename = 'train_rgb/RGB_' + fileNumber
depthFilename = 'train_rgb/DEPTH_' + fileNumber
textureMap(lidarFilename, jointFilename, rgbFilename, depthFilename)