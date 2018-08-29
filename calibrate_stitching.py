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
from videoStitch.homography import Stitcher
import yaml
from utilities import *
from PIL import Image



def doit_with_opencv(VIDS, N=7):
    #CALCULATE HOMOGRAPHY
    imgs = []
    for v in VIDS:               
        v.set(1,N)
        ret, i = v.read()
        if ret:
            imgs.append(i)
        else:
            break
    stitcher = Stitcher()
    (top, top_vis), Mtop = stitcher.getHomography([imgs[0],imgs[2]], 'horizontal', showMatches=True)
    (bot, bot_vis), Mbot = stitcher.getHomography([imgs[1],imgs[3]], 'horizontal', showMatches=True) 
    (result, vis), Mtotal = stitcher.getHomography([top,bot], 'vertical', showMatches=True) 
    
    if not None in [Mtop, Mbot, Mtotal]:
        H = {'topRow':Mtop[1], 'bottomRow': Mbot[1], 'final':Mtotal[1]}
        with open(SAVEAS + '.yml', 'w') as outfile:
            yaml.dump(H, outfile, default_flow_style=False)
        Image.fromarray(result).save(SAVEAS+'.png')
    return H


def calculate_stitching(VIDS, ITER):
    
    imgs = []
    
    for v in VIDS:
        img, (f, t) = v.get_next_frame()
        imgs.append(img)
    
    stitcher = Stitcher()
    (top, top_vis), Mtop = stitcher.getHomography([imgs[0],imgs[2]], 'horizontal', showMatches=True)
    (bot, bot_vis), Mbot = stitcher.getHomography([imgs[1],imgs[3]], 'horizontal', showMatches=True) 
    (result, vis), Mtotal = stitcher.getHomography([top,bot], 'vertical', showMatches=True) 
    
    if not None in [Mtop, Mbot, Mtotal]:
        H = {'topRow':Mtop[1], 'bottomRow': Mbot[1], 'final':Mtotal[1]}
        with open(SAVEAS + '_' + str(ITER) +'.yml', 'w') as outfile:
            yaml.dump(H, outfile, default_flow_style=False)
        Image.fromarray(result).save(SAVEAS+'_' + str(ITER) +'.png')
        return H
    else:
        return 0    


  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/', help='path to videos')
    parser.add_argument('--handle', type=str, required=False, default='',
                                    help='unique catchall for videos to be stitched. timestamp works well')
    #parser.add_argument('--saveas', type=str, required=False, default='notAssigned', help='output filename')
    args = parser.parse_args()
    
    #SEARCH_FILES = '/media/recnodes/recnode_2mfish/stitch10000_20180503_160717/undistorted/stitch*'
    SEARCH_FILES = slashdir(args.dir) + '*' +  args.handle + '*undistorted'
    

    #Camera orientations before renovation:
    [tl, bl, tr, br] = ['21990445',
                        '21990447',
                        '21990449',
                        '21990443']
    """
    #Camera orientations after renovation (August 2018):
    """ #Current settings as of 180827
    [tl, bl, tr, br] = ['21990447',
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
        SAVEAS = '/home/dan/videoStitch/calibration/homography/homography_' + ts
    else:
        SAVEAS = '/home/dan/videoStitch/calibration/homography/' + args.saveas
       

    
    #H = doit_with_opencv(VIDEOS)
    
    for i in range(TOP_LEFT.frame_count-1):
        H = calculate_stitching(VIDEOS, i)
    
    print 'done'
    print H
 
    
            
        
        
