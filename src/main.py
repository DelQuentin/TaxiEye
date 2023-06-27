from Controller import Controller
from Segmentation import segmentation
import time,winsound,cv2
from PIL import ImageGrab
import numpy as np

DEBUG = True
SIM = False

if __name__ == "__main__":

    # ===== INITIALIZE =====
    stop = False
    t = 0
    toFollow = 0
    # Controller
    if SIM:
        controller = Controller()
        time.sleep(5)
        winsound.Beep(1000,100)
        # Init commands ready for use
        controller.initSim()
        controller.idle()
        # Release Park and enable NWS then start taxi
        controller.action("PARK",0)
        controller.action("NWS",0)
    winsound.Beep(1000,100)
    # Horizontal Limit of view Extration
    viewLimit = 370
    # ======================


    # ====== RECORDING ======
    vidcap = cv2.VideoCapture('../Recordings/Taxi_clear.mp4')
    # ======================


    # ====== PROCESS ======
    run = True
    while run:
        if SIM:
            image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(320,31,1600+320,900+31))),cv2.COLOR_RGB2BGR)
        else:
            # ====== RECORDING =======
            success,image = vidcap.read()
            if not success:
                run = False
                break
            # image = cv2.imread('../Recordings/test.png')

        # ====== CROSSING SEGMENTATION ======
        seg_data = segmentation(image,viewLimit,DEBUG)

        # ====== LINES MATHCING ======
        # ToDo

        # ====== CROSSING NAVIGATION ======
        # ToDo
        # POC
        # if len(seg_data) != 0:
        #     toFollow = abs(seg_data[0])
        #     for line in seg_data:
        #         if abs(line)<= abs(toFollow):
        #             toFollow = line
        
        # ====== CONTROL SYSTEM ======
        gain = 0.1
        command = gain * toFollow
        if DEBUG: print(command)
        if SIM:
            controller.action("THROTTLE",-0.2)
            controller.action("RUDDER",command)
        
        # ====== CYCLE END ======
        if DEBUG: 
            cv2.imshow("Camera Feed",cv2.resize(image,(640,360)))
            key = cv2.waitKey(1)
            if key == 27:
                run = False
        t += 1
        if t >= 5000 and SIM:
            run = False
    # ====================== 

    winsound.Beep(1000,100)