from SLAM import finalSLAM

#name of the testing filepath
fileNumber = '0'
lidarFilePath = 'data/train_lidar' + fileNumber
jointFilePath = 'data/train_joint' + fileNumber

finalSLAM(lidarFilePath,jointFilePath)