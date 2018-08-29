"""
Undistorts images based on calibration matrices calculated with calibrate.py.


Calibration files are automatically selected based on the timestamp of the video, selecting the latest calibration time that precedes the time of recording.

Returns undistorted videos in a new directory named by appending "_undistorted" to the input.

D.Bath 2018
"""


import numpy as np
import cv2
import glob
import yaml
from utilities import *
import imgstore
import os
import shutil

class Undistort:

    def __init__(self, store):
        self.vidfile = store.filename
        self.video_timestamp = '_'.join(self.vidfile.split('/')[-1].split('.')[0].rsplit('_',2)[1:])
        self.startTime = getTimeFromTimeString(self.video_timestamp)
        self.camSerial = store.filename.split('.')[1]
        self.calibrationFile = self.selectCalibrationFile()
        
        k, d = self.loadCameraConfig(self.calibrationFile)
        
        self.cameraMatrix = np.array(k) #UMAT
        self.cameraDistortion = np.array(d) #UMAT
        
        print "\nDewarping video: ", self.vidfile, "using calibration: ", self.calibrationFile
        
        return
    
    def selectCalibrationFile(self):
        """
        returns the calibration file from the correct camera 
        from the most recent calibration before the video was created.
        """
        fileList = []
        times = []
        for x in glob.glob('/home/dan/fishMAD/camera_calibrations/*.yaml'):
            fileList.append(x)
            times.append(getTimeFromTimeString(x.split('/')[-1].split('.')[0].rsplit('_',1)[0]))
        df = pd.DataFrame({'filename':fileList, 'times':times})
        calTime = df[df.times < self.startTime].max()['filename'].split('/')[-1].rsplit('_',1)[0]
        
        return '/home/dan/fishMAD/camera_calibrations/' + calTime + '_' + self.camSerial + '.yaml'
        
    def loadCameraConfig(self, CALIBRATION):
        with open(CALIBRATION) as f:
            loadeddict = yaml.load(f)

        mtxloaded = loadeddict.get('camera_matrix')
        distloaded = loadeddict.get('dist_coeff')
        
        return mtxloaded, distloaded

    def undistort(self, img): 
        if not hasattr(self, 'newcamera'):
            h,w = img.shape[:2]
            self.newcamera, roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.cameraDistortion, (w,h), 0) 
            self.newcamera = self.newcamera#UMAT
        
        return cv2.undistort(img, self.cameraMatrix, self.cameraDistortion, None, self.newcamera)#.get() downloads it from the graphics card #UMAT

def doit(DIR, HANDLE):
    for x in glob.glob(slashdir(DIR) + '*' + HANDLE + '*/metadata.yaml'):
        inStore = imgstore.new_for_filename(x)
        UND = Undistort(inStore)
        img, (frame_number, frame_timestamp) = inStore.get_next_image()
        
        newdir = x.rsplit('/',1)[0]+'_undistorted'
        if os.path.exists(newdir):
            shutil.rmtree(newdir)
        os.mkdir(newdir)
        outStore = imgstore.new_for_format('jpg', mode='w', 
                    basedir=newdir, 
                    imgshape=img.shape, 
                    imgdtype=img.dtype,
                    chunksize=500)
        
        for i in range(inStore.frame_count-1):
            try:
                outStore.add_image(UND.undistort(img), frame_number, frame_timestamp) 
                img, (frame_number, frame_timestamp) = inStore.get_next_image()
            except:
                print "failed at frame: ", i , "of", inStore.frame_count, inStore.frame_max   
   
    return    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--handle', type=str, required=True, help='unique identifier for videos to be undistorted')
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/', help='directory to search for files')
    args = parser.parse_args()

    for x in glob.glob(slashdir(args.dir) + '*' + args.handle + '*/metadata.yaml'):
        inStore = imgstore.new_for_filename(x)
        UND = Undistort(inStore)
        
        newdir = x.rsplit('/',1)[0]+'_undistorted'
        if os.path.exists(newdir):
            shutil.rmtree(newdir)
        os.mkdir(newdir)
        outStore = imgstore.new_for_format('jpg', mode='w', 
                    basedir=newdir, 
                    imgshape=inStore.image_shape, 
                    imgdtype='uint8',
                    chunksize=500)
        
        for i in range(inStore.frame_count-1):
            try:
                img, (frame_number, frame_timestamp) = inStore.get_next_image()
                outStore.add_image(UND.undistort(img), frame_number, frame_timestamp) 
            except:
                print "failed at frame: ", i , "of", inStore.frame_count, inStore.frame_max   
    
    
          
            
            
            
    print "done."      

