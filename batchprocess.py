import os.path as op
import re

import cv2
import numpy as np
import pandas as pd

import imgstore
from imgstore import new_for_filename, new_for_format
from imgstore.util import ensure_color

from stitchup import Stitcher, StoreAligner, new_window

import datetime, glob
import os, sys
import yaml


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
        for x in glob.glob(os.path.expanduser('~/stitchup/camera_calibrations/*.yaml')):
            fileList.append(x)
            times.append(getTimeFromTimeString(x.split('/')[-1].split('.')[0].rsplit('_',1)[0]))
        df = pd.DataFrame({'filename':fileList, 'times':times})
        calTime = df[df.times < self.startTime].max()['filename'].split('/')[-1].rsplit('_',1)[0]
        
        return os.path.expanduser('~/stitchup/camera_calibrations/' + calTime + '_' + self.camSerial + '.yaml')
        
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

from contextlib import contextmanager

@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target
        
def getTimeFromTimeString(string=None):
    if string == None:
        return datetime.datetime.now()
    elif '_' in string:
        return datetime.datetime.strptime(string, "%Y%m%d_%H%M%S")
    else:
        return datetime.datetime.strptime(time.strftime("%Y%m%d", time.localtime()) + '_' + string, "%Y%m%d_%H%M%S")

def getTimeStringFromTime(TIME=None):
    if TIME == None:
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")
    else:
        return datetime.datetime.strftime(TIME,"%Y%m%d_%H%M%S")


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = 'X', stdoutpos = 1):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print '\r%s |%s| %s%% %s   ' % (prefix, bar, percent, suffix),'\r'*stdoutpos
    # Print New Line on Complete
    if iteration == total: 
        print '\n'
    return
    
    
def get_store_camera_serial(store):
    try:
        return str(store.user_metadata['camera_serial'])
    except KeyError:
        m = re.match(r".*\.(\d+).*/metadata\.yaml$", store.full_path)
        return str(m.groups()[0])



def create_stitched_video_from_undistorted(fnString):


    stores = (fnString + '.21990443_undistorted',
              fnString + '.21990445_undistorted',
              fnString + '.21990447_undistorted',
              fnString + '.21990449_undistorted')

    cams_stores = {}
    for fn in stores:
        store = new_for_filename(op.join(BASE_DATA, fn))
        print store.full_path
        cams_stores[get_store_camera_serial(store)] = store    

    sorted_stores = [cams_stores[i] for i in best_order]
    
    aligned = StoreAligner(*sorted_stores)
    aligned.extract_common_frames(StoreAligner.MISSING_POLICY_DROP)
    if not os.path.exists(BASE_DATA + fnString + '.stitched'):
        os.mkdir(BASE_DATA + fnString + '.stitched')

    out = imgstore.new_for_format( 'avc1/mp4', mode='w', 

                basedir=BASE_DATA + fnString,
                imgshape=s.panorama_shape,
                imgdtype='uint8',
                chunksize=500)
    for n, (fn, imgs) in enumerate(aligned.iter_imgs()):
        
        
        ok, img = s.stitch_images(*[ensure_color(i) for i in imgs])
        assert ok

        out.add_image(img, fn, 0)

        print n
    out.close()
    
    return


def create_stitched_video_from_scratch(fnString, pos):


    stores = (fnString + '.21990443',
              fnString + '.21990445',
              fnString + '.21990447',
              fnString + '.21990449')

    cams_stores = {}
    for fn in stores:
        store = new_for_filename(op.join(BASE_DATA, fn))
        print store.full_path
        nFrames = store.frame_count
        cams_stores[get_store_camera_serial(store)] = store    

    sorted_stores = [cams_stores[i] for i in best_order]

    undistortions = [Undistort(i) for i in sorted_stores]
    
    aligned = StoreAligner(*sorted_stores)
    aligned.extract_common_frames(StoreAligner.MISSING_POLICY_DROP)
    if not os.path.exists(fnString + '.stitched'):
        os.mkdir(fnString + '.stitched')

    out = imgstore.new_for_format( 'avc1/mp4', mode='w', 

                basedir=fnString + '.stitched',
                imgshape=s.panorama_shape,
                imgdtype='uint8',
                chunksize=500)
    for n, (imgs, (fn,ts)) in enumerate(aligned.iter_imgs(return_times=True)):
        
        _imgs = []
        for i in  range(len(undistortions)):
            _imgs.append(undistortions[i].undistort(imgs[i]))

        #with silence_stdout():        
        ok, img = s.stitch_images(*[ensure_color(i) for i in _imgs])
        assert ok

        out.add_image(img, fn, ts)

        printProgressBar(n,nFrames, prefix='Stitching progress:', stdoutpos=pos) 
    out.close()
    
    return


if __name__ == "__main__":

    import argparse
    import glob
    from multiprocessing import Process
    
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--calibdir', type=str, required=False, default='/media/recnodes/recnode_2mfish/',
			 help='path to directory containing calibration videos')
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/',
			 help='path to directory containing videos to be stitched')
    parser.add_argument('--handle', type=str, required=True, help='unique identifier that marks the files to use for calibration. Ideally use a custom handle for your project. Or for a single file, pass the timestamp, ie "20180808_153229".')
    
    parser.add_argument('--date', type=str, required=False, default='current', help='if processing old data, provide a date to determine which calibration to use. Default is to most recent calibration.'  )
    args = parser.parse_args()

    
    BASE_CALIB = args.calibdir
    BASE_DATA = args.dir
    
    HANDLE = args.handle.split(',')
    DIRECTORIES = args.dir.split(',')




    # I dont actually think the order matters, it's most important that the order
    # is conserved between calibration and stitching
    best_order = ('21990449','21990447','21990445','21990443')##('21990443', '21990447', '21990445', '21990449')
    
    if 0:#args.date == 'current':

        calibs = ('stitch_20190715_145513.21990443_undistorted',
                  'stitch_20190715_145513.21990445_undistorted',
                  'stitch_20190715_145513.21990447_undistorted',
                  'stitch_20190715_145513.21990449_undistorted')
    elif 0:#getTimeFromTimeString(args.date + '_000000') >= getTimeFromTimeString('20190715_000000'):

        calibs = ('stitch_20190715_145513.21990443_undistorted',
                  'stitch_20190715_145513.21990445_undistorted',
                  'stitch_20190715_145513.21990447_undistorted',
                  'stitch_20190715_145513.21990449_undistorted')

    elif 0:#getTimeFromTimeString(args.date + '_000000') >= getTimeFromTimeString('20190620_000000'):
    
        calibs = ('stitch_dry_26c_20190624_091343.21990443_undistorted',
                  'stitch_dry_26c_20190624_091343.21990445_undistorted',
                  'stitch_dry_26c_20190624_091343.21990447_undistorted',
                  'stitch_dry_26c_20190624_091343.21990449_undistorted')
    else:
        calibs = ('stitch09_20180910_165817.21990443_undistorted',
                  'stitch09_20180910_165817.21990447_undistorted',
                  'stitch09_20180910_165817.21990445_undistorted',
                  'stitch09_20180910_165817.21990449_undistorted')
    
    cams_imgs = {}

    # load the first frame for the calibration
    for fn in calibs:
        with new_for_filename(op.join(BASE_CALIB, fn)) as store:
            camera_serial = get_store_camera_serial(store)
            img, _ = store.get_image(frame_number=None, exact_only=True, frame_index=0)
            cams_imgs[camera_serial] = ensure_color(img)

    sorted_imgs = [cams_imgs[i] for i in best_order]

    s = Stitcher(use_gpu=True,
                 estimator_type="homography",
                 matcher_type="affine",
                 warp_type="plane")


    # s.enable_exposure_compensation('gain_blocks')
    # s.enable_seam_finding('gc_color')

    #with silence_stdout():
    ok = s.load_calibration(*sorted_imgs)
    assert ok

    new_window('panorama', shape=s.panorama_shape)


    s.enable_blending('feather', 1)




    for x in range(len(DIRECTORIES)):
        if DIRECTORIES[x][-1] != '/':
            DIRECTORIES[x] += '/'
    
    filelist = []        
    for term in HANDLE:
        for DIR in DIRECTORIES:
            for vDir in glob.glob(DIR + '*' + term + '*.21990443'):
                if os.path.getsize(vDir + '/000000.mp4') < 1e1: #FIXME return to 1e6
                    continue #don't stitch dark videos.
                elif os.path.exists(vDir.rsplit('.',1)[0] + '.stitched'):
                    continue
                elif "undistorted" in vDir:
                    continue #FIXME
                    #create_stitched_video_from_undistorted(vDir.rsplit('.',1)[0])
                else:
                    print vDir
                    create_stitched_video_from_scratch(vDir.rsplit('.',1)[0],0)
                    #filelist.append(vDir.rsplit('.',1)[0])#create_stitched_video_from_scratch(vDir.rsplit('.',1)[0])
    """
    threadcount=0
    for filenum in range(len(filelist)):
        threadcount += 1
        p = Process(target=create_stitched_video_from_scratch, args=(filelist[filenum],threadcount))
        p.start()
        #print "processing: ", vDir
        
        if p.is_alive():
            if (threadcount >= 3) or (filenum == len(fileList)):
                threadcount = 0
                p.join()
    """


