import cv2
import pysky360

algorithm = pysky360.WeightedMovingVariance()

video_file = "Dahua-20220901-184734.mp4"
capture = cv2.VideoCapture(video_file)

while not capture.isOpened():
    flag, frame = capture.read()

    if flag:
        cv2.imshow('video', frame)
        #pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        #pos_frame = capture.get(cv2.CV_CAP_PROP_POS_FRAMES)
        pos_frame = capture.get(1)
        #print str(pos_frame)+" frames"
        
        img_output = algorithm.apply(frame)
        #img_bgmodel = algorithm.getBackgroundModel()
        
        cv2.imshow('img_output', img_output)
        #cv2.imshow('img_bgmodel', img_bgmodel)

    else:
        #capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        #capture.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        #capture.set(1, pos_frame-1)
        #print "Frame is not ready"
        cv2.waitKey(1000)
        break
    
    if 0xFF & cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()