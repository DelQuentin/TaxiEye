from ACS import ACS
from Avionics import Avionics
from Controller import Controller
from Map import Map
from Navigator import Navigator
from segmentation_func import segmentation
import time,winsound,cv2
from PIL import ImageGrab
import numpy as np
import random,os

DEBUG = True
SIM = True
AVIONICS = True

if __name__ == "__main__":

    # ===== INITIALIZE =====
    stop = False
    t = 0
    
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

    # ====== RECORDING ======
    vidcap = cv2.VideoCapture('../Recordings/P2toF1.mp4')

    # Horizontal Limit of view Extration
    viewLimit = [1050,140]

    # ====== AVIONICS ======
    avionics = Avionics()

    # ===== AIRCRAFT CONTROL SYSTEM =====
    acs = ACS(7)

    # ====== MAP INIT ======
    map = Map('Creech')
    map_img = map.map_image()

    # ====== NAVIGATOR INIT ======
    path = ["P2","A2_P2_3","A3","A3_P3_1","A4","A4_F1_2","F1"]
    curr_line = path[0]
    curr_dir = 0
    nav = Navigator(curr_line,curr_dir,path,map)

    # ====== PROCESS ======
    run = True
    while run:
        if DEBUG: print("\n ===== Debug =====")
        # ====== IMAGE SOURCE =======
        if SIM:
            image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0,0,1920,1080))),cv2.COLOR_RGB2BGR)
        else:
            success,image = vidcap.read()
            if not success:
                run = False
                break

        # ====== AVIONICS ======
        hdg,spd,pos_x,pos_y = avionics.extract(image,DEBUG)
        if DEBUG: print("hdg:{} spd:{} X:{} Y:{}".format(hdg,spd,pos_x,pos_y))

        # Generate GPS noise
        sig = 3 # Standard Error of Sensor Fusion GPS/INS
        gps = [pos_x + random.gauss(0,sig),pos_y + random.gauss(0,sig)]
        if DEBUG: print("GPS: {} - {}".format(gps[0],gps[1]))

        # ====== CROSSING SEGMENTATION ======
        seg_data,dbg_image = segmentation(image,viewLimit,DEBUG)
        center = np.shape(dbg_image)[1]/2

        # ====== CROSSING MAP MATCHING AND NAVIGATION ======
        if AVIONICS:
            curr_line,labels,line_to_follow,pos_corr = nav.run_with_avionics(seg_data,gps,hdg,DEBUG)
        else:
            curr_line,labels,line_to_follow = nav.run_no_avionics(seg_data,DEBUG)
        
        # ====== CONTROL SYSTEM ======
        rudder, throttle, brakes = acs.run(line_to_follow,center,spd,DEBUG)
        if DEBUG: print("Throttle:{} Rudder:{} Brakes:{}".format(throttle,rudder,brakes))
        if SIM:
            controller.action("THROTTLE",throttle)
            controller.action("RUDDER",rudder)
            controller.action("",brakes)
        
        # ====== DEBUG DISPLAY ======
        if DEBUG:
            cv2.imshow("Camera Feed",cv2.resize(image,(640,360)))
            # ===== CAMERA DEBUG VIEW =====
            # Display Labels of the matching
            for l in range(len(seg_data)):
                if labels[l]:
                    x = int((seg_data[l][-1][0]+seg_data[l][1][0])/2)
                    y = int((seg_data[l][-1][1]+seg_data[l][1][1])/2)
                    dbg_image = cv2.putText(dbg_image,labels[l],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,2)
            # Display the path chosen by the navigator
            for l in range(len(line_to_follow)-1):
                dbg_image = cv2.line(dbg_image,[line_to_follow[l][0]-20,line_to_follow[l][1]],[line_to_follow[l+1][0]-20,line_to_follow[l+1][1]],(255,255,255),2)
                dbg_image = cv2.line(dbg_image,[line_to_follow[l][0]+20,line_to_follow[l][1]],[line_to_follow[l+1][0]+20,line_to_follow[l+1][1]],(255,255,255),2)
            # Display Avionics Data
            dbg_image = cv2.putText(dbg_image,"HDG : "+str(hdg),(10,545),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2)
            dbg_image = cv2.putText(dbg_image,"SPD : "+str(spd),(10,560),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2)
            dbg_image = cv2.putText(dbg_image,"C_X : "+str(pos_x),(10,575),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2)
            dbg_image = cv2.putText(dbg_image,"C_Y : "+str(pos_y),(10,590),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2)
            # Display Control Data
            # Rudder
            dbg_image = cv2.putText(dbg_image,"Rudder : "+str(int(rudder*100)),(920,530),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2)
            dbg_image = cv2.rectangle(dbg_image,(1000,540),(1000+int(rudder*100),580),(255,0,0),-1)
            dbg_image = cv2.rectangle(dbg_image,(900,540),(1100,580),(255,255,255),2)
            # Throttle & Brakes
            dbg_image = cv2.putText(dbg_image,"Throttle : "+str(int(throttle*100)),(930,510),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2)
            dbg_image = cv2.putText(dbg_image,"Brakes : "+str(int(brakes*100)),(930,490),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2)
            dbg_image = cv2.rectangle(dbg_image,(1060,530),(1100,530-int(throttle*200)),(255,0,0),-1)
            dbg_image = cv2.rectangle(dbg_image,(1060,330),(1100,330+int(brakes*200)),(0,0,255),-1)
            dbg_image = cv2.rectangle(dbg_image,(1060,330),(1100,530),(255,255,255),2)
            cv2.imshow("System Image Data",dbg_image)
            cv2.imwrite(''.join([os.getcwd(),'/exports/camera-',str(t),'.png']),dbg_image)

            # ===== MAP DEBUG VIEW =====
            map_img_step = map_img.copy()
            map_img_step = map.point_on_image(map_img_step,gps[0],gps[1],2,(0,0,255),2)
            map_img_step = map.point_on_image(map_img_step,pos_corr[0],pos_corr[1],2,(255,0,0),2)
            cv2.imshow("System Map Data",map_img_step)
            cv2.imwrite(''.join([os.getcwd(),'/exports/map-',str(t),'.png']),map_img_step)
            
            key = cv2.waitKey(1)
            if key == 27:
                run = False
            # Stop at each step for intensive debug
        
        # ===== Cycle End =====
        t += 1
        if t >= 5000 and SIM:
            run = False

    # ===== Data Analysis =====
    if DEBUG: 
        nav.plot_data(False)
        acs.display_recordings(True)

    # ===== Process End =====
    winsound.Beep(1000,100)