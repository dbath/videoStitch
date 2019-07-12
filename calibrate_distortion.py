"""
Generates calibration matrices for undistorting images. 

Ideal input is a video, or set of videos, in imgstore format with low framerate (1 or 2Hz) containing checkerboard images. Call the script with a unique identifier that tags only the video or videos to run. This is usually the timestamp of the video recording.

Output is a set of calibration matrices that are saved by default in the videoStitch git repository, although save location can be customized.  Image points and a summary figure are also saved in the directory of the input video.

D. Bath 2018

"""


import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import imgstore
import random
import os


def showFeatures(img, df, fn):
    plt.imshow(img)

    plt.scatter(df['x'], df['y'], color='r', s=1)
    plt.axis('off')
    if fn == 'nosave':
        plt.show()
    else:
        plt.savefig(fn, layout='Tight')
    return

def coords(a):
    # where a is a cornerSubPix Array
    xs = []
    ys = []
    for x in range(len(a)):
        xs.append(a[x][0])
        ys.append(a[x][1])
    df = pd.DataFrame({'x':xs, 'y':ys})
    return df
    
def getPointsList(a):
    # where a is a cornerSubPix Array
    return pd.DataFrame([a[x][0] for x in range(len(a))]) 
     
def calibrate(store, CHECKERSHAPE, DESTFILE):

    """
    Pass a imgstore object, tuple representing checker shape, and a place to save the calibration file.
    """
    
    checkerShape = CHECKERSHAPE
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerShape[0]*checkerShape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerShape[0],0:checkerShape[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    found = 0
    errorcount = 0
    FRAME_NUMBERS = store.get_frame_metadata()['frame_number']
    random.shuffle(FRAME_NUMBERS)
    TARGET_NFRAMES = 40
    MAX_ERRORS = 40
    if store.frame_count < (TARGET_NFRAMES + MAX_ERRORS):
        TARGET_NFRAMES = store.frame_count / 3
        MAX_ERRORS = store.frame_count/3
    #while (found <= 50):  #  can be changed to whatever number you like to choose
    for k in FRAME_NUMBERS:
        if found > TARGET_NFRAMES:
            break
        else:
            try:
                img, (framenum, timestamp) = store.get_image(k)
            except:
                img, (framenum, timestamp) = store.get_next_image()
            try:
                
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(img, checkerShape,None)
                #
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
                    corners2 = cv2.cornerSubPix(img,corners,checkerShape,(5,5),criteria)
                    imgpoints.append(corners2)
                    found += 1
                    lastImg = img
                    #showFeatures(img, pd.DataFrame(getPointsList(corners2) ,columns=['x','y']), 'nosave')
            except:
                errorcount +=1
                if errorcount > MAX_ERRORS:
                    print "Could not generate full calibration file: ", DESTFILE.split('/')[-1]
                    break            
            
                
    df = pd.concat([getPointsList(imgpoints[x]) for x in range(len(imgpoints))])
    df.columns = ['x','y']

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL)

    # It's very important to transform the matrix to list.

    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

    
    with open(DESTFILE, "w") as f:
        yaml.dump(data, f)
    
    df.to_pickle(store.filename + '/imgPoints.pickle')
    showFeatures(lastImg, df, store.filename + '/featureList.png')    
    print 'Calibration complete: ', DESTFILE.split('/')[-1]

    
    
    return 




if __name__ == "__main__":

    import argparse
    import glob
    from utilities import *
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/',
			 help='path to directory containing checker vids')
    parser.add_argument('--handle', type=str, required=True, help='unique identifier that marks the files to use for calibration. Ideally use the timestamp of the recording, ie "20180808_153229".')
    parser.add_argument('--checkersize', type=str, required=False, default='12x12', help='size of checkerboard, default is 11x11')
    parser.add_argument('--saveas', type=str, required=False, default='notDefined', help='name for calibration, including date time string, ex: 20180404_123456')

                
    args = parser.parse_args()
    
    CHECKERSIZE = tuple([int(k) for k in args.checkersize.split('x')])

    for vid in glob.glob(slashdir(args.dir) + '*' + args.handle + '*/metadata.yaml'):
        if "undistorted" in vid: #skip already processed videos
            continue
        inStore = imgstore.new_for_filename(vid)    
    
        if args.saveas == 'notDefined':
            DATETIME = '_'.join(vid.split('/')[-2].split('.')[0].rsplit('_',2)[1:])
            SERIAL = inStore.user_metadata['camera_serial']
            SAVE_AS = '_'.join([DATETIME,SERIAL])
        else:
            SAVE_AS = '_'.join([args.saveas, inStore.user_metadata['camera_serial']])
        print SAVE_AS
        calibrate(inStore, CHECKERSIZE, os.path.expanduser('~/videoStitch/calibrations/distortion/'+SAVE_AS+'.yaml')) 











