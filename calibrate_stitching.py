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
<<<<<<< HEAD
from homography import Stitcher
import yaml
from utilities import *
from PIL import Image
import imgstore


def sho(img):
    plt.imshow(img)
    plt.show()
    return

def sbs(imglist, axes_off=False):
    fig = plt.figure()
    for i in range(len(imglist)):
        ax = plt.subplot(1,len(imglist), i+1)
        plt.imshow(imglist[i])
        if axes_off:
            plt.axis('off')
    plt.show()
    return
=======
from videoStitch.homography import Stitcher
import yaml
from utilities import *
from PIL import Image


>>>>>>> 955ab2b6c17676546eacbeb7b8081df7f4f070f7

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


<<<<<<< HEAD

def manual_imglist(myString):
    imgs = []
    for d in ['3','5','7','9']:
        imgs.append(cv2.imread(myString + '.2199044'+ d + '_undistorted/000000/000000.jpg'))
    return imgs
    
    
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect
	
	
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped




def getCorners(img):
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(
            ix, iy)

        coords.append((ix, iy))

        if len(coords) == 4:
            fig.canvas.mpl_disconnect(cid)
            plt.close('all')
        return coords

    coords = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    coords = np.array(coords)
    coords[coords < 4.0] = 0.0
    for dim in img.shape[:2]:
        coords[abs(coords - dim) < 4.0] = dim
    return coords
        
=======
>>>>>>> 955ab2b6c17676546eacbeb7b8081df7f4f070f7
  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/', help='path to videos')
    parser.add_argument('--handle', type=str, required=False, default='',
                                    help='unique catchall for videos to be stitched. timestamp works well')
<<<<<<< HEAD
    parser.add_argument('--saveas', type=str, required=False, default='notAssigned', help='output filename')
=======
    #parser.add_argument('--saveas', type=str, required=False, default='notAssigned', help='output filename')
>>>>>>> 955ab2b6c17676546eacbeb7b8081df7f4f070f7
    args = parser.parse_args()
    
    #SEARCH_FILES = '/media/recnodes/recnode_2mfish/stitch10000_20180503_160717/undistorted/stitch*'
    SEARCH_FILES = slashdir(args.dir) + '*' +  args.handle + '*undistorted'
    
<<<<<<< HEAD
    """
=======

>>>>>>> 955ab2b6c17676546eacbeb7b8081df7f4f070f7
    #Camera orientations before renovation:
    [tl, bl, tr, br] = ['21990445',
                        '21990447',
                        '21990449',
                        '21990443']
    """
    #Camera orientations after renovation (August 2018):
<<<<<<< HEAD
     #Current settings as of 180827
    [tr, br, tl, bl] = ['21990447',
=======
    """ #Current settings as of 180827
    [tl, bl, tr, br] = ['21990447',
>>>>>>> 955ab2b6c17676546eacbeb7b8081df7f4f070f7
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
 
    
            
        
        
