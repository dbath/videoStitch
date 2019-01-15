import os.path as op
import re

import cv2
import numpy as np

import imgstore
from imgstore import new_for_filename, new_for_format
from imgstore.util import ensure_color

from stitchup import Stitcher, StoreAligner, new_window

from undistort import Undistort

def get_store_camera_serial(store):
    try:
        return str(store.user_metadata['camera_serial'])
    except KeyError:
        m = re.match(r".*\.(\d+).*/metadata\.yaml$", store.full_path)
        return str(m.groups()[0])


TESTING = False


BASE_CALIB = '/media/recnodes/recnode_2mfish/'

# I dont actually think the order matters, it's most important that the order
# is conserved between calibration and stitching
best_order = ('21990443', '21990447', '21990445', '21990449')

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

ok = s.load_calibration(*sorted_imgs)
assert ok

#new_window('panorama', shape=s.panorama_shape)

if TESTING:
    ok, img = s.stitch_images(*sorted_imgs)
    cv2.imshow('panorama', img)
    cv2.waitKey(0)
    raise SystemExit

#s.enable_blending('feather', 1)

BASE_DATA = '/media/recnodes/recnode_2mfish/'

def create_stitched_video_from_undistorted(path):

    DIRECTORY, fnString = path.rsplit('/',1)
    DIRECTORY = DIRECTORY + '/'
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
    if not os.path.exists(DIRECTORY + fnString + '.stitched'):
        os.mkdir(DIRECTORY + fnString + '.stitched')
    """
    out = new_for_format('avc1/mp4', DIRECTORY + fnString + '.stitched/metadata.yaml',
                         imgshape=s.panorama_shape,
                         imgdtype=np.uint8)
    """
    out = imgstore.new_for_format( 'avc1/mp4', mode='w', 

                basedir=DIRECTORY + fnString + '.stitched',
                imgshape=s.panorama_shape,
                imgdtype='uint8',
                chunksize=500)
    for n, (imgs, (fn, ts)) in enumerate(aligned.iter_imgs(return_times=True)):
        
        
        ok, img = s.stitch_images(*[ensure_color(i) for i in imgs])
        assert ok


        out.add_image(img, fn, ts)

        print n
    out.close()
    
    return


def create_stitched_video_from_scratch(path):

    DIRECTORY, fnString = path.rsplit('/',1)
    DIRECTORY = DIRECTORY + '/'
    stores = (fnString + '.21990443',
              fnString + '.21990445',
              fnString + '.21990447',
              fnString + '.21990449')

    cams_stores = {}
    for fn in stores:
        store = new_for_filename(op.join(BASE_DATA, fn))
        print store.full_path
        cams_stores[get_store_camera_serial(store)] = store    

    sorted_stores = [cams_stores[i] for i in best_order]

    undistortions = [Undistort(i) for i in sorted_stores]
    
    aligned = StoreAligner(*sorted_stores)
    aligned.extract_common_frames(StoreAligner.MISSING_POLICY_DROP)
    if not os.path.exists(DIRECTORY + fnString + '.stitched'):
        os.mkdir(DIRECTORY + fnString + '.stitched')
    """
    out = new_for_format('avc1/mp4', DIRECTORY + fnString + '.stitched/metadata.yaml',
                         imgshape=s.panorama_shape,
                         imgdtype=np.uint8)
    """
    out = imgstore.new_for_format( 'avc1/mp4', mode='w', 

                basedir=DIRECTORY + fnString + '.stitched',
                imgshape=s.panorama_shape,
                imgdtype='uint8',
                chunksize=500)
    for n, (imgs, (fn, ts)) in enumerate(aligned.iter_imgs(return_times=True)):
        
        _imgs = []
        for i in  range(len(undistortions)):
            _imgs.append(undistortions[i].undistort(imgs[i]))
        
        ok, img = s.stitch_images(*[ensure_color(i) for i in _imgs])
        assert ok

        out.add_image(img, fn, ts)

    out.close()
    
    return


if __name__ == "__main__":

    import argparse
    import glob
    from utilities import *
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=False, default='/media/recnodes/recnode_2mfish/',
			 help='path to directory containing checker vids')
    parser.add_argument('--handle', type=str, required=True, help='unique identifier that marks the files to use for calibration. Ideally use the timestamp of the recording, ie "20180808_153229".')
      
    args = parser.parse_args()
    HANDLE = args.handle.split(',')
    DIRECTORIES = args.dir.split(',')
    for x in range(len(DIRECTORIES)):
        if DIRECTORIES[x][-1] != '/':
            DIRECTORIES[x] += '/'
            
    for term in HANDLE:
        for DIR in DIRECTORIES:
            for vDir in glob.glob(DIR + '*' + term + '*.21990443'):
                if os.path.exists(vDir.rsplit('.',1)[0] + '.stitched'):
                    continue
                elif "undistorted" in vDir:
                    continue #FIXME
                    create_stitched_video_from_undistorted(vDir.rsplit('.',1)[0])
                else:
                    create_stitched_video_from_scratch(vDir.rsplit('.',1)[0])

