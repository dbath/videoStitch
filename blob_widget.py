"""

Created in 2020 by Dan Bath
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def binarize(img):
    #input image (RGB or greyscale), returns binarized image
    
    if len(img.shape) > 2:
        img = img[:,:,0]
    

    #anything brighter than background is a reflection. Assuming the median value
    # is background, replace those with more than 40% of the median brightness:
    
    THRESH = np.median(img)*0.9
    
    ret, th2 = cv2.threshold(img,THRESH, THRESH ,cv2.THRESH_TRUNC)
    
    # use adaptive threshold
    th3 = cv2.adaptiveThreshold(th2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,35,5)
    
    #clean up by erosion and dilation
    kernel = np.ones((5,5),np.uint8)
    binary = cv2.erode(th3.copy(), kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=2)
    
    return binary


def get_contours(binary_img):
    global IMG_H, IMG_W
    (IMG_H, IMG_W) = binary_img.shape[0:2]

    px_per_cm = float(IMG_W/IMAGE_WIDTH) 

    MIN_AREA = MINMAX_BLOB_AREA[0]*(px_per_cm**2)
    MAX_AREA = MINMAX_BLOB_AREA[1]*(px_per_cm**2)


    # first get all the contours
    if float(cv2.__version__[0]) < 4.0:
        _, contours, hierarchy = cv2.findContours(binary_img.copy(), 
                                           cv2.RETR_CCOMP, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    elif float(cv2.__version__[0]) >= 4.0:
        contours, hierarchy = cv2.findContours(binary_img.copy(), 
                                           cv2.RETR_CCOMP, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    # now filter by size
    objects = []
    for c in contours:
        a = cv2.contourArea(c)
        if MIN_AREA < a < MAX_AREA:
            objects.append(c)
    return objects

def markObjects(img, objects):
    
    for c in objects:
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(img, (cx, cy), 
                   np.around(0.002*IMG_W).astype(int),(0,255,0), -1)
    cv2.putText(img,   str(len(objects)) + ' objects', (int(0.5*IMG_W), 
                int(0.5*IMG_W)), cv2.FONT_HERSHEY_SIMPLEX,2, (255,255,255), 
                np.around(0.002*IMG_W).astype(int))  
    return img
    

def createBackgroundFromVideo(vid):
    """takes cv2 video instance, returns image with modal pixel values"""
    framecount = int(vid.get(7))
    skipFrames = int(np.round(framecount/11)) 
    frameList = range(1,framecount, skipFrames)
    for frame in frameList:
        vid.set(1, frame)
        ret, _img = vid.read()    
        if ret == True:
            i = _img[:,:,0]
            if frame == 1:
                img_array = i
            else:
                img_array = np.dstack((img_array, i))
        else:
            vid.release()
            break  
    vid.set(1,1)
    try: 
        import scipy.stats as stats
        return stats.mode(img_array, axis=2)[0][:,:,0]
    except:
        print("unable to use scipy.stats.mode. using max instead")
        return np.max(img_array, axis=2)

def getMaxCount(vid, bkg="Not-assigned"):
    """takes cv2 video instance, returns max number of blobs. 
       we use the max because occlusions are common. the frame with 
       the max count likely has the fewest occlusions"""
    framecount = int(vid.get(7))
    skipFrames = int(np.round(framecount/7)) 
    frameList = range(1,framecount, skipFrames)
    maxCount = 0
    maxFrame = 0
    for frame in frameList:
        vid.set(1, frame)
        ret, _img = vid.read()  
        if bkg != "Not-assigned":
            if len(bkg.shape) > 2:
                bkg = bkg[:,:,0]
            _img = _img[:,:,0]-bkg  
        if ret == True:
            count = len(get_contours(binarize(_img)))
            if count > maxCount:
                maxCount = count
                maxFrame = frame
        else:
            vid.release()
            break  
    
    vid.set(1, maxFrame)
    ret, _img = vid.read()
    vid.set(1,1)
    return maxCount ,   _img, maxFrame
    


if __name__ == "__main__":

    from tkinter import *
    from tkinter import filedialog as fd 
    from tkinter import simpledialog as sd

    def setFN():
        global filename
        filename = fd.askopenfilename(title="select image or video",
                                      initialdir='/Users/bathd/videoStitch/blob')
        print('file:', filename)
        return filename

    def setBKG():
        global bkg
        bkg = fd.askopenfilename(title="select image or video",
                                      initialdir='/Users/bathd/videoStitch/blob')
        print('background:', bkg)
        return bkg   

        
    # use Tkinter to open explorer
    gui = Tk() 
    gui.title('Blob counter')
    gui.geometry('400x400') 
    
    #enter video file
    filename = StringVar()
    filename.set('asdf')
    lblFN = Label(gui, text='Input File').grid(row = 0, column = 0, 
                                               padx = 0, pady = 10)
    butFN = Button(gui, text='Select', command=setFN)
    butFN.grid(row = 0, column = 1, padx = 0, pady = 10)

    #optionally enter bkg. default uses adaptive thresholds
    bkg = StringVar()
    bkg.set('Not-assigned')
    bkg = bkg.get()
    lblBKG = Label(gui, text='Background File').grid(row = 1, column = 0, 
                                                     padx = 0, pady = 10)
    butBKG = Button(gui,text='Select', command=setBKG)
    butBKG.grid(row = 1, column = 1,  padx = 0, pady = 10)

    #set width of image in cm
    width = StringVar()
    width.set('400')
    lblW = Label(gui, text='image width (cm)').grid(row = 2, column = 0, 
                                                    padx=0, pady = 10)
    entW = Entry(gui, textvariable=width).grid(row=2, column=1)
 
    min_area = StringVar()
    min_area.set('0.5')
    lblminA = Label(gui, text='min blob area').grid(row = 3, column = 0,  
                                                    padx = 0, pady = 10)
    entminA = Entry(gui, textvariable=min_area).grid(row=3, column=1)

    max_area = StringVar()
    max_area.set('17')
    lblmaxA = Label(gui, text='max blob area').grid(row = 4, column = 0,  
                                                    padx = 0, pady = 10)
    entmaxA = Entry(gui, textvariable=max_area).grid(row=4, column=1)
    
    button = Button(text = "Go", command = gui.destroy).grid(row=5, column=1)
    
    
    gui.mainloop()
    
    
    IMAGE_WIDTH = int(width.get())
    
    MINMAX_BLOB_AREA = (float(min_area.get()), float(max_area.get()))
 
    print(filename, '\n', bkg)
    
    if filename.split('.')[-1] == 'mp4': 
        vid = cv2.VideoCapture(filename)
        if bkg != "Not-assigned":
            _bkg = createBackgroundFromVideo(vid)
        else:
            _bkg = "Not-assigned"
        count, img, maxframe = getMaxCount(vid, bkg=_bkg)
        print("Maximum count (", str(count),") found in frame: ", str(maxframe))
        blobs = get_contours(binarize(img))
    
    else:
        img = cv2.imread(filename)
        if bkg != "Not-assigned":
            _bkg = cv2.imread(bkg)
            img = img-_bkg
        blobs = get_contours(binarize(img))
        count = len(blobs)
    print("detected ", str(count), "objects.")
    plt.imshow(markObjects(img, blobs))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)    
    plt.show()

    

