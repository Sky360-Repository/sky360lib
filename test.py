import cv2
import pysky360

algorithm = pysky360.WeightedMovingVariance()

video_file = "Dahua-20220901-184734.mp4"
capture = cv2.VideoCapture(video_file)

while not capture.isOpened():
    capture = cv2.VideoCapture(video_file)
    cv2.waitKey(1000)
    print("Wait for the header")

while True:
    flag, frame = capture.read()

    if flag:
        cv2.imshow('video', frame)
        pos_frame = capture.get(1)
        img_output = algorithm.apply(frame)
        
        cv2.imshow('img_output', img_output)

    else:
        cv2.waitKey(1000)
        break
    
    if 0xFF & cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()