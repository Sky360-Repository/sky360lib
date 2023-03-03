import cv2
import pysky360

camera = pysky360.QHYCamera()
#algorithm = pysky360.WeightedMovingVariance()
algorithm = pysky360.Vibe()

parameters = algorithm.getParameters()
threshold = parameters.getThreshold()
print("threshold: " + str(threshold))

camera.open('')

while True:
    frame = camera.getFrame(True)

    frame = cv2.resize(frame, (1024, 1024))
    cv2.imshow('video', frame)
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    img_output = algorithm.apply(greyFrame)
    
    cv2.imshow('img_output', img_output)

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