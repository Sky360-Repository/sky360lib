import cv2
import pysky360

algorithm = pysky360.WeightedMovingVariance()
# algorithm = pysky360.WeightedMovingVarianceHalide()
# algorithm = pysky360.WeightedMovingVarianceCuda()

video_file = "Dahua-20220901-184734.mp4"
capture = cv2.VideoCapture(video_file)

while not capture.isOpened():
    capture = cv2.VideoCapture(video_file)
    cv2.waitKey(1000)
    print("Wait for the header")

while True:
    flag, frame = capture.read()

    if flag:
        frame = cv2.resize(frame, (1024, 1024))
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