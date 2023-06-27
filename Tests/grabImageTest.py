from PIL import ImageGrab
import cv2
import numpy as np

while True:
    screen = np.array(ImageGrab.grab(bbox=(320,31,1600+320,900+31)))
    cv2.imshow('Python Window', screen)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break