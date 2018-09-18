#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:02:08 2018

@author: dan
"""

from utilities import *
import pandas as pd
import imgstore
import functools



def stitchRL(imageB,imageA,H):
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    return result

def stitchTB(imageB,imageA,H):
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0] + imageB.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    return result

def selectHomographyFile(vidTime):
    vidTime = getTimeFromTimeString(vidTime)
    fileList = []
    times = []
    for x in glob.glob('/home/dan/videoStitch/calibrations/homography/*.yml'):
        fileList.append(x)
        times.append(getTimeFromTimeString(x.split('/')[-1].split('.')[0].split('_',1)[1]))
    df = pd.DataFrame({'filename':fileList, 'times':times})
    return df[df.times < vidTime].max()['filename']
    
  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/', help='path to videos')
    parser.add_argument('--handle', type=str, required=False, default='',
                                    help='unique catchall for videos to be stitched. timestamp works well')
    parser.add_argument('--saveas', type=str, required=False, default='notAssigned', help='output filename')
    args = parser.parse_args()
    
    #SEARCH_FILES = '/media/recnodes/recnode_2mfish/stitch10000_20180503_160717/undistorted/stitch*'
    SEARCH_FILES = slashdir(args.dir) + '*' +  args.handle + '*_undistorted/metadata.yaml'

    """    
    #Camera orientations before renovation:
    [tl, bl, tr, br] = ['21990445',
                        '21990447',
                        '21990449',
                        '21990443']

    #Camera orientations after renovation (August 2018):
    """ 
    #Current settings as of 180827
    [tl, bl, tr, br] = ['21990447',
                        '21990449',
                        '21990443',
                        '21990445']

    
    
    for x in glob.glob(SEARCH_FILES):
        ID = x.split('/')[-2].split('.')[-1]
        print ID
        if br in ID:
            BOTTOM_RIGHT = imgstore.new_for_filename(x)
        elif tl in ID:
            TOP_LEFT = imgstore.new_for_filename(x)
        elif bl in ID:
            BOTTOM_LEFT = imgstore.new_for_filename(x)
        elif tr in ID:
            TOP_RIGHT = imgstore.new_for_filename(x)
 
    
    if args.saveas == 'notAssigned':
        SAVEAS = TOP_LEFT.filename.rsplit('.',1)[0] + '_stitched'
    else:
        SAVEAS = slashdir(args.dir) + args.saveas 
        
    VIDEOS = [TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT]

    
    VIDEO_TIME = '_'.join(TOP_LEFT.filename.split('/')[-1].split('.')[0].split('_')[-2:])
    HOMOGRAPHY_FILE = selectHomographyFile(VIDEO_TIME)
    #h = yaml.load(open('stitch/homography_20180827_150058.yml')) #for dev and debugging
    h = yaml.load(open(HOMOGRAPHY_FILE))

    IMG_SHAPE = (TOP_LEFT.image_shape[0] + TOP_RIGHT.image_shape[0],
                 TOP_LEFT.image_shape[1] + BOTTOM_LEFT.image_shape[1])

    if os.path.exists(SAVEAS):
        shutil.rmtree(SAVEAS)
    os.mkdir(SAVEAS)
    outStore = imgstore.new_for_format('jpg', mode='w', 
                basedir=SAVEAS, 
                imgshape=IMG_SHAPE, 
                imgdtype='uint8',
                chunksize=500)

    store_fns = [store.get_frame_metadata()['frame_number'] for store in VIDEOS]
    
    common = functools.reduce(np.intersect1d, store_fns)


    for i in common:#range(TOP_LEFT.frame_min, TOP_LEFT.frame_max):
        imgs = []
        for vid in VIDEOS:
            try:
                img, (f, t) = vid.get_image(i)
            except:
                img, (f, t) = vid.get_next_image()
            
            imgs.append(img)
        
        top = stitchRL(imgs[0], imgs[2], h['topRow'])
        bottom = stitchRL(imgs[1], imgs[3], h['bottomRow'])
        final = stitchTB(top, bottom, h['final'])
        
        outStore.add_image(final, f, t)
    outStore.close()
    print "Done."
    
    """

    while TOP_LEFT.isOpened():
        framenum = TOP_LEFT.get(1)
        imgs = []
        for v in VIDS:               
            v.set(1,framenum)
            ret, i = v.read()
            if ret:
                imgs.append(i)
            else:
                break
        if len(imgs) != 4:
            break
        top = stitchRL(imgs[0], imgs[2], h['topRow'])
        bottom = stitchRL(imgs[1], imgs[3], h['bottomRow'])
        final = stitchTB(top, bottom, h['final'])

    """
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    