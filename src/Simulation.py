# =============== Libraries Import ===============
import winsound, time, cv2, random, os
import numpy as np
from PIL import ImageGrab

# ================= Files Import =================
from segmentation_func import segmentation
from Controller import Controller
from Navigator import Navigator
from Avionics import Avionics
from ACS import ACS
from Map import Map

# =============== Class Definition ===============
class Simulation:
    """Simulation Class which oversees the process of simulation of the Taxiway Navigation System with simulators, and especially coded here for Digital Combat Simulator® by Eagle Dynamics."""

    def __init__(self, simulator: str ,source: str, map_name: str, path: list, camera_settings: list, spd_tgt: int, rudder_pid: list, deviation_feedback_params: list, seg_scale: int, slid_win: list) -> None:
        """Simulation Class constructor

        ----------
            Parameters
        simulator : str 
            -> Indicates which flight simulator will be used as image source and simulation engine.
        source: str 
            -> Indicates if the simulation will be using a recording (give path here) or real-time data (using 'Realtime').
        map: str 
            -> Map name of where the simulation will take place.
        path: str 
            -> Path on the taxiway that should be followed (has to be consistent with flight simulator set up or recording).
        spd_tgt: int 
            -> Speed target during taxi (in knots for DCS)
        deviation_feedback_params: dict
            -> mu and sigma (in list) parameters for guasiian distribution used for weighted sum in feedback generation for rudder control loop
        seg_scale: int
            -> Pixels per meter in the homographic transform used for segmentation purposes
        slid_win: list
            -> Dimensions of the sliding window in the segmentation process [left<->right,up<->down] in meters
        """

        # Simulation Parameters
        self.sim = simulator
        self.src = source
        self.path = path
        self.cam = camera_settings
        self.step = 0
        self.seg_scale = seg_scale
        self.slid_win = slid_win

        # Video Source Initialisation
        if self.src == 'Realtime':
            self.ctr = Controller()
            time.sleep(5)
            winsound.Beep(1000,100)
            # Init commands ready for use
            self.ctr.initSim()
            self.ctr.idle()
            # Release Park and enable NWS then start taxi
            self.ctr.action("PARK",0)
            self.ctr.action("NWS",0)
            winsound.Beep(1000,100)
        else:
            self.vidcap = cv2.VideoCapture(self.src)

        # Systems Initialisation
        self.acs = ACS(spd_tgt,rudder_pid,deviation_feedback_params)
        self.avi = Avionics()
        self.map = Map(map_name)
        self.map_img = self.map.map_image()
        self.nav = Navigator(path[0],0,self.path,self.map)
    
    def run(self,debug: bool):
        """Simulation Class constructor

        ----------
            Parameters
        debug : bool 
            -> Indicates if debug information will be displayed

        ----------
            Returns
        end_flag : bool
            -> Indicates if the simulation reached the end goal
        """

        # ===== Begin =====
        if debug: print("\n ===== Simulation Step {} =====".format(self.step))
        end_flag = False

        # ===== Retrive Camera Feed =====
        camera_feed = []
        if self.src == 'Realtime':
            camera_feed = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0,0,1920,1080))),cv2.COLOR_RGB2BGR)
        else:
            success,camera_feed = self.vidcap.read()
            if not success:
                end_flag = True
                return end_flag
        
        # ===== Run Systems =====
        # Run Avionics
        hdg,spd,pos_x,pos_y = self.avi.extract(camera_feed)
        if debug: print("hdg:{} spd:{} X:{} Y:{}".format(hdg,spd,pos_x,pos_y))

        # Generate GPS signal
        sig = 3 # Standard Error of Sensor Fusion GPS/INS
        gps = [pos_x + random.gauss(0,sig),pos_y + random.gauss(0,sig)]
        if debug: print("GPS: {} - {}".format(gps[0],gps[1]))

        # Run Taxiway Navigation System
        seg_data,dbg_image = segmentation(camera_feed,self.cam,self.seg_scale,self.slid_win)
        center = np.shape(dbg_image)[1]/2
        labels,line_to_follow,pos_corr,end_flag = self.nav.run_with_avionics(seg_data,gps,hdg) # A version without avionics exists in case the avionics system is not availbale (not using DCS, or disbling it on purpose)

        # Run Aircraft Control System
        rudder, throttle, brakes = self.acs.run(line_to_follow,center,spd)
        if debug: print("Throttle:{} Rudder:{} Brakes:{}".format(throttle,rudder,brakes))

        # Send commands to Simulator
        if self.src == 'Realtime':
            if end_flag:
                self.ctr.idle()
            else:
                self.ctr.action("THROTTLE",throttle)
                self.ctr.action("RUDDER",rudder)
                self.ctr.action("",brakes)

        # ===== Debug Windows =====
        if debug:
            # === Camera Feed ===
            cv2.imshow('Camera Feed',cv2.resize(camera_feed,(640,360)))
            # === Vision Data ===
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
            # cv2.imwrite(''.join([os.getcwd(),'/exports/camera-',str(self.step),'.png']),dbg_image)

            # === Map Data ===
            # map_img_step = self.map_img.copy()
            # map_img_step = self.map.point_on_image(map_img_step,gps[0],gps[1],2,(0,0,255),2)
            # map_img_step = self.map.point_on_image(map_img_step,pos_corr[0],pos_corr[1],2,(255,0,0),2)
            # cv2.imshow("System Map Data",map_img_step)
            # cv2.imwrite(''.join([os.getcwd(),'/exports/map-',str(self.step),'.png']),map_img_step)
            cv2.waitKey(0)

        # ===== End =====
        cv2.waitKey(1)
        self.step += 1
        return end_flag

    def plot_data(self):
        self.acs.display_recordings(True)