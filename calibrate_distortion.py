"""
Generates calibration matrices for undistorting images. 

Ideal input is a video, or set of videos, in imgstore format with low framerate (1 or 2Hz) containing checkerboard images. Call the script with a unique identifier that tags only the video or videos to run. This is usually the timestamp of the video recording.

Output is a set of calibration matrices that are saved by default in the fishmad git repository, although save location can be customized.  Image points and a summary figure are also saved in the directory of the input video.

D. Bath 2018

"""


import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import pandas as pd



def showFeatures(img, df, fn):
     plt.imshow(img)

     plt.scatter(df['x'], df['y'], color='r', s=2)
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
    while (found < 90):  #  can be changed to whatever number you like to choose
    
        img, (framenum, timestamp) = store.get_next_frame()
        
        if framenume%3 ==0: #every nth frame
            try:
                gray = img[:,:,0]

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, checkerShape,None)

                #
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners2)
                    found += 1
                    lastImg = img
                    #showFeatures(img, pd.DataFrame(getPointsList(corners2) ,columns=['x','y']), 'nosave')
            except:
                errorcount +=1
                if errorcount > 100:
                    print "Could not generate full calibration file: ", DESTFILE.split('/')[-1]
                    break            
    
                
    df = pd.concat([getPointsList(imgpoints[x]) for x in range(len(imgpoints))])
    df.columns = ['x','y']
    df.to_pickle(store.filename + '/imgPoints.pickle')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # It's very important to transform the matrix to list.

    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

    showFeatures(lastImg, df, store.filename + '/featureList.png')
    
    with open(DESTFILE, "w") as f:
        yaml.dump(data, f)
        
    print 'Calibration complete: ', DESTFILE.split('/')[-1]

    
    
    return 




if __name__ == "__main__":

    import argparse
    import glob
    from utilities import *
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/',
			 help='path to directory containing checker vids')
    parser.add_argument('--timestamp', type=str, required=True, help='unique identifier that marks the files to use for calibration. Ideally use the timestamp of the recording, ie "20180808_153229".'
    parser.add_argument('--checkersize', type=str, required=False, default='6x6', help='size of checkerboard, default is 6x6')
    parser.add_argument('--saveas', type=str, required=False, default='notDefined', help='name for calibration, including date time string, ex: 20180404_123456')

                
    args = parser.parse_args()
    
    CHECKERSIZE = tuple([int(k) for k in args.checkersize.split('x')])

    for vid in glob.glob(slashdir(args.dir) + '*' + args.handle + '*/metadata.yaml'):
        inStore = imgstore.new_for_filename(vid)    
    
        if args.saveas == 'notDefined':
            DATETIME = '_'.join(vid.split('/')[-2].split('.')[0].rsplit('_',2)[1:])
            SERIAL = vid.split('.')[-2].split('/')[0]
            SAVE_AS = '_'.join([DATETIME,SERIAL])
        else:
            SAVE_AS = '_'.join([args.saveas, vid.split('.')[-2]])

        calibrate(inStore, CHECKERSIZE, '/home/dan/fishMAD/camera_calibrations/'+SAVE_AS+'.yaml') 











