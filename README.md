# videoStitch


18/09/2018 - This repository is under construction and is not yet prepared for public distribution. Parameters such as camera serials and file paths are still hard-coded.  Until this repository is finished, issues/bug fix requests may go unanswered. Sorry for the inconvenience and thanks for your understanding. 




# Instructions:

In order to stitch videos, we must first remove distortion caused by the four cameras, then use feature matching to define homography between the images.

1. Calibrate undistortion

Generate a video from each camera recorded at ~1fps with a checkerboard pattern moving around the image. Give each video a common handle so we know which videos to calibrate together. Call the calibration protocol and specify a date and time at which the calibration was recorded. This date/time will determine which recordings will use this calibration.

      python calibrate_distortion.py --handle='your_handle_here' --saveas='myDATE_myTIME'
      
2. Undistort videos

Batch process undistortion of all videos containing a specified handle by calling: 

    python undistort.py --handle='your_handle_here'

3. Calibrate stitching

Place non-repetitive, feature-rich images in the overlapping regions of the space, aligned in the plane to be calibrated. Record a few frames from all cameras, again specifying a handle to link them together (you can also simply use the timestamps). calibrate_stitching.py generates a .yaml file containing all necessary homography matrices, as well as a sample stitched image located in videoStitch/calibrations/homography. Similar to undistortion, specify "saveas" with a date and time at which the calibration was recorded. This date/time will determine which recordings will use this calibration.


    python calibrate_stitching.py --handle='your_handle_here' --saveas='20180830_060000'
    
    
4. Stitch

Stitch a video by calling stitch.py and specifying the video to be stitched. For best results, use the timestamp in the filename as the handle. Also specify a new filename with which to save the stitched video. 

    python stitch.py --handle='20180901_145200' --saveas='/full/path/to/directory/myStitchedVideo'
    

    
    
    

# Installation: 

clone this repository and install with 
       
       python setup.py install

      
      

    
