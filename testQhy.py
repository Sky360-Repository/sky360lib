import cv2
import pysky360

camera = pysky360.QHYCamera()
#algorithm = pysky360.WeightedMovingVariance()
algorithm = pysky360.Vibe()

parameters = algorithm.getParameters()
threshold = parameters.getThreshold()
print("threshold: " + str(threshold))

cv2.namedWindow("BGS", cv2.WINDOW_NORMAL);
cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL);

cv2.resizeWindow("BGS", (1024, 1024))
cv2.resizeWindow("Live Video", (1024, 1024));

camera.open('')
camera.setControl(pysky360.ControlParam.Exposure, 40000, False)

while True:
    # greyFrame = camera.getFrame(False)
    # frame = camera.debayerImage(greyFrame)
    frame = camera.getFrame(True)

    cv2.imshow('Live Video', frame)
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    bgsMask = algorithm.apply(greyFrame)
    cv2.imshow('BGS', bgsMask)

    keyPressed = cv2.waitKey(10) & 0xFF
    if keyPressed == 27:
        break
    elif keyPressed == ord('+'):
        threshold += 5
        parameters.setThreshold(threshold)
        print("threshold: " + str(threshold))
    elif keyPressed == ord('-'):
        threshold -= 5
        parameters.setThreshold(threshold)
        print("threshold: " + str(threshold))

cv2.destroyAllWindows()