"""
This script calculates the homography values for a group of 4 cameras. 

Input: a directory with undistorted videos containing busy, non-repetitive 
    images with sufficient overlapping regions.

Output: three sets of homography values: top row, bottom row, and top-to-bottom
a stitched image
"""
 
 
 
import numpy as np
import imutils
import cv2
from homography import Stitcher
import yaml
from utilities import *
from PIL import Image
import imgstore
import os


def sho(img):
    plt.imshow(img)
    plt.show()
    return

def sbs(imglist, axes_off=False):
    """
    displays images side by side for comparison while optimizing
    """
    fig = plt.figure()
    for i in range(len(imglist)):
        ax = plt.subplot(1,len(imglist), i+1)
        plt.imshow(imglist[i])
        if axes_off:
            plt.axis('off')
    plt.show()
    return


def calculate_stitching(VIDS):
    
    imgs = []
    
    for v in VIDS:
        img, (f, t) = v.get_next_image()
        imgs.append(img)


    stitcher = Stitcher()

    top, Htop, topShape, topROI = getStitch(imgs[2], imgs[0], 'horizontal')
    bot, Hbot, botShape, botROI = getStitch(imgs[3], imgs[1], 'horizontal') 
   
    
    #FINAL STITCH
    result, Htotal, finalShape, ROI = getStitch(bot, top, 'vertical')
    
    #try:
    H = {'topRow':np.asarray(Htop).tolist(), 
         'bottomRow': np.asarray(Hbot).tolist(), 
         'final':np.asarray(Htotal).tolist(), 
         'ROI' : np.asarray(ROI).tolist(),
          'topShape':np.asarray(topShape).tolist(), 
          'botShape':np.asarray(botShape).tolist(), 
          'finalShape':np.asarray(finalShape).tolist()}
    with open(SAVEAS  + '.yml', "w") as outfile:
        yaml.dump(H, outfile, default_flow_style=False)
    result = result[ROI[0]:ROI[1],ROI[2]:ROI[3]]
    Image.fromarray(result).save(SAVEAS +'.png')
    return H
    #except:
    #    return 0    



def manual_imglist(myString):
    imgs = []
    for d in ['3','5','7','9']:
        imgs.append(cv2.imread(myString + '.2199044'+ d + '_undistorted/000000/000000.jpg'))
    return imgs
    


def getStitch(WARP, WARPTO, DIRECTION):
    MAP, H, shape = stitcher.getHomography([WARPTO, WARP], DIRECTION)
    res = cv2.warpAffine(WARP, H, (MAP.shape[1],MAP.shape[0]))
    cop = res.copy() 
    cop[0:WARPTO.shape[0], 0:WARPTO.shape[1]] = WARPTO
    res = np.maximum(res, cop)
    res = res.astype(np.uint8)
    #CROP:
    (y, x) = res[:,:].nonzero()
    ROI = (y.min(), y.max(), x.min(), x.max())
    return res, H, shape, ROI



def stitch(WARP, WARPTO, shape, H, ROI=None):
    res = cv2.warpAffine(WARP, H, (shape[1], shape[0]))
    cop = res.copy() 
    cop[0:WARPTO.shape[0], 0:WARPTO.shape[1]] = WARPTO
    res = np.maximum(res, cop)
    res = res.astype(np.uint8)
    if ROI != None:
        return res[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    else:    
        return res




        
  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/', help='path to videos')
    parser.add_argument('--handle', type=str, required=False, default='',
                                    help='unique catchall for videos to be stitched. timestamp works well')
    parser.add_argument('--saveas', type=str, required=False, default='notAssigned', help='output filename representing the date at which to use this calibration, in format "YYMMDD_HHMMSS", example "20180830_060000" will use this calibration for files recorded after 30th of August 2018 at 6am.')
    args = parser.parse_args()
    
    #SEARCH_FILES = '/media/recnodes/recnode_2mfish/stitch10000_20180503_160717/undistorted/stitch*'
    SEARCH_FILES = slashdir(args.dir) + '*' +  args.handle + '*undistorted'
    
    """
    #Camera orientations before renovation:
    [tl, bl, tr, br] = ['21990445',
                        '21990447',
                        '21990449',
                        '21990443']
    """
    #Camera orientations after renovation (August 2018):
     #Current settings as of 180827
    [tr, br, tl, bl] = ['21990447',
                        '21990449',
                        '21990443',
                        '21990445']

    
    
    for x in glob.glob(SEARCH_FILES):
        ID = x.split('.')[-1]
        print ID
        if br in ID:
            BOTTOM_RIGHT = imgstore.new_for_filename(x+'/metadata.yaml')
        elif tl in ID:
            TOP_LEFT = imgstore.new_for_filename(x+'/metadata.yaml')
        elif bl in ID:
            BOTTOM_LEFT = imgstore.new_for_filename(x+'/metadata.yaml')
        elif tr in ID:
            TOP_RIGHT = imgstore.new_for_filename(x+'/metadata.yaml')

            
    VIDEOS = [TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT]


    if args.saveas == 'notAssigned':
        ts = '_'.join(TOP_LEFT.filename.split('/')[-1].split('.')[0].split('_')[-2:])
        SAVEAS = '/home/dan/videoStitch/calibrations/homography/homography_' + ts
    else:
        SAVEAS = '/home/dan/videoStitch/calibrations/homography/' + args.saveas
       

    
    stitcher = Stitcher()
    
    H = calculate_stitching(VIDEOS)
    
    print 'done'
    print H
 
    
            
        
        
