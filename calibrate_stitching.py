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

def rotateAndWarp(warp, warpto, rotateCode='norotation'):
    if len(warp.shape) == 2:
        warp = cv2.cvtColor(warp, cv2.COLOR_GRAY2BGRA)
    elif warp.shape[2] == 3:
        warp = cv2.cvtColor(warp, cv2.COLOR_BGR2BGRA)
    if len(warpto.shape) == 2:
        warpto = cv2.cvtColor(warpto, cv2.COLOR_GRAY2BGRA)
    elif warpto.shape[2] == 3:
        warpto = cv2.cvtColor(warpto, cv2.COLOR_BGR2BGRA)
    
    if not rotateCode == 'norotation':
        warp = cv2.rotate(warp, rotateCode)
        warpto = cv2.rotate(warpto, rotateCode)
    
    (kpsC, featuresC) = detectAndDescribe(warpto)
    (kpsBR, featuresBR) = detectAndDescribe(warp) 
    (matches, H, status) = matchKeypoints(kpsBR, kpsC, featuresBR, featuresC, 0.75,4.0)
    SHAPE = (warp.shape[1] + warpto.shape[1], warp.shape[0]+warpto.shape[0])
    result = cv2.warpPerspective(warp, H, SHAPE) 
    
    #result[0:warpto.shape[0], 0:warpto.shape[1]] = warpto   
    if rotateCode == cv2.ROTATE_180:
        result = cv2.rotate(result, cv2.ROTATE_180)
    elif rotateCode == cv2.ROTATE_90_CLOCKWISE:
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotateCode == cv2.ROTATE_90_COUNTERCLOCKWISE:
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        
    return result, H, SHAPE

def calculate_stitching(VIDS, ITER):
    
    imgs = []
    
    for v in VIDS:
        img, (f, t) = v.get_next_image()
        imgs.append(img)


    #resize centre image to approximately match corner vids
    small = cv2.resize(imgs[4], (int(imgs[4].shape[1]/3.0), int(imgs[4].shape[0]/3.0)))

    (kpsC, featuresC) = detectAndDescribe(small)

    #warp each image to the centre plane
    Rtl, Htl, ShapeTL = rotateAndWarp(imgs[0], small, cv2.ROTATE_180)
    Rbl, Hbl, ShapeBL = rotateAndWarp(imgs[1], small, cv2.ROTATE_90_COUNTERCLOCKWISE)
    Rtr, Htr, ShapeTR = rotateAndWarp(imgs[2], small, cv2.ROTATE_90_CLOCKWISE)
    Rbr, Hbr, ShapeBR = rotateAndWarp(imgs[3], small)

    
    stitcher = Stitcher()

    top, Htop, topShape, topROI = getStitch(Rtr, Rtl, 'horizontal')
    bot, Hbot, botShape, botROI = getStitch(Rbr, Rbl, 'horizontal') 
    # I GUESS I NEED TO UNDISTORT TOP AND BOTTOM HERE. UGHHHHHHH>
    
    
    #FINAL STITCH
    result, Htotal, finalShape, ROI = getStitch(bot, top, 'vertical')
    
    try:
        H = {'topRow':Htop, 'bottomRow': Hbot, 'final':Htotal, 'ROI' : ROI,
              'topShape':topShape, 'botShape':botShape, 'finalShape':finalShape}
        with open(SAVEAS + '_' + str(ITER) +'.yml', "w") as outfile:
            yaml.dump(H, outfile, default_flow_style=False)
        result = result[ROI[0]:ROI[1],ROI[2]:ROI[3]]
        Image.fromarray(result).save(SAVEAS+'_' + str(ITER) +'.png')
        return H
    except:
        return 0    



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


def getStitch(WARP, WARPTO, DIRECTION):
    MAP, H, shape = stitcher.getHomography([WARPTO, WARP], DIRECTION)
    res = cv2.warpPerspective(WARP, H, (MAP.shape[1],MAP.shape[0]))
    cop = res.copy() 
    cop[0:WARPTO.shape[0], 0:WARPTO.shape[1]] = WARPTO
    res = np.maximum(res, cop)
    res = res.astype(np.uint8)
    #CROP:
    (y, x) = res[:,:,3].nonzero()
    ROI = (y.min(), y.max(), x.min(), x.max())
    return res, H, shape, ROI



def stitch(WARP, WARPTO, shape, H, ROI=None):
    res = cv2.warpPerspective(WARP, H, (shape[1], shape[0]))
    cop = res.copy() 
    cop[0:WARPTO.shape[0], 0:WARPTO.shape[1]] = WARPTO
    res = np.maximum(res, cop)
    res = res.astype(np.uint8)
    if ROI != None:
        return res[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    else:    
        return res



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
        
  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/', help='path to videos')
    parser.add_argument('--handle', type=str, required=False, default='',
                                    help='unique catchall for videos to be stitched. timestamp works well')
    parser.add_argument('--saveas', type=str, required=False, default='notAssigned', help='output filename')
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
    [tr, br, tl, bl, centre] = ['21990447',
                        '21990449',
                        '21990443',
                        '21990445',
                        '40012623']

    
    
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
        elif centre in ID:
            CENTRE = imgstore.new_for_filename(x+'/metadata.yaml')
            
    VIDEOS = [TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT, CENTRE]


    if args.saveas == 'notAssigned':
        ts = '_'.join(TOP_LEFT.filename.split('/')[-1].split('.')[0].split('_')[-2:])
        SAVEAS = '/home/dan/videoStitch/calibrations/homography/homography_' + ts
    else:
        SAVEAS = '/home/dan/videoStitch/calibrations/homography/' + args.saveas
       

    
    #H = doit_with_opencv(VIDEOS)
    
    for i in range(TOP_LEFT.frame_count-1):
        H = calculate_stitching(VIDEOS, i)
    
    print 'done'
    print H
 
    
            
        
        
