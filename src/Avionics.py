import cv2
import numpy as np

class Avionics:

    def __init__(self):
        self.hdg = 0
        self.spd = 0
        self.x = 0
        self.y = 0
        self.digits = self.load_digits()
    
    def extract(self,img):
        frame_sz = [8,10]
        hdg_frame = [[676,1055],[669,1055],[662,1055]]
        spd_frame = [[544,1055],[537,1055],[530,1055]]
        X_frame = [[681,1068],[674,1068],[667,1068],[660,1068],[653,1068]]
        Y_frame = [[753,1068],[746,1068],[739,1068],[732,1068]]

        # Recognition
        hdg_x = self.recognition(img[hdg_frame[0][1]:hdg_frame[0][1]+frame_sz[1],hdg_frame[0][0]:hdg_frame[0][0]+frame_sz[0]])
        hdg_xx = self.recognition(img[hdg_frame[1][1]:hdg_frame[1][1]+frame_sz[1],hdg_frame[1][0]:hdg_frame[1][0]+frame_sz[0]])
        hdg_xxx = self.recognition(img[hdg_frame[2][1]:hdg_frame[2][1]+frame_sz[1],hdg_frame[2][0]:hdg_frame[2][0]+frame_sz[0]])
        spd_x = self.recognition(img[spd_frame[0][1]:spd_frame[0][1]+frame_sz[1],spd_frame[0][0]:spd_frame[0][0]+frame_sz[0]])
        spd_xx = self.recognition(img[spd_frame[1][1]:spd_frame[1][1]+frame_sz[1],spd_frame[1][0]:spd_frame[1][0]+frame_sz[0]])
        spd_xxx = self.recognition(img[spd_frame[2][1]:spd_frame[2][1]+frame_sz[1],spd_frame[2][0]:spd_frame[2][0]+frame_sz[0]])
        X_x = self.recognition(img[X_frame[0][1]:X_frame[0][1]+frame_sz[1],X_frame[0][0]:X_frame[0][0]+frame_sz[0]])
        X_xx = self.recognition(img[X_frame[1][1]:X_frame[1][1]+frame_sz[1],X_frame[1][0]:X_frame[1][0]+frame_sz[0]])
        X_xxx = self.recognition(img[X_frame[2][1]:X_frame[2][1]+frame_sz[1],X_frame[2][0]:X_frame[2][0]+frame_sz[0]])
        X_xxxx = self.recognition(img[X_frame[3][1]:X_frame[3][1]+frame_sz[1],X_frame[3][0]:X_frame[3][0]+frame_sz[0]])
        X_xxxxx = self.recognition(img[X_frame[4][1]:X_frame[4][1]+frame_sz[1],X_frame[4][0]:X_frame[4][0]+frame_sz[0]])
        Y_x = self.recognition(img[Y_frame[0][1]:Y_frame[0][1]+frame_sz[1],Y_frame[0][0]:Y_frame[0][0]+frame_sz[0]])
        Y_xx = self.recognition(img[Y_frame[1][1]:Y_frame[1][1]+frame_sz[1],Y_frame[1][0]:Y_frame[1][0]+frame_sz[0]])
        Y_xxx = self.recognition(img[Y_frame[2][1]:Y_frame[2][1]+frame_sz[1],Y_frame[2][0]:Y_frame[2][0]+frame_sz[0]])
        Y_xxxx = self.recognition(img[Y_frame[3][1]:Y_frame[3][1]+frame_sz[1],Y_frame[3][0]:Y_frame[3][0]+frame_sz[0]])

        # Calculate Numbers
        self.hdg = hdg_x + 10*hdg_xx + 100*hdg_xxx
        self.spd = spd_x + 10*spd_xx + 100*spd_xxx
        self.x = X_x + 10*X_xx + 100*X_xxx + 1000*X_xxxx + 10000*X_xxxxx
        self.y = Y_x + 10*Y_xx + 100*Y_xxx + 1000*Y_xxxx

        return self.hdg,self.spd,self.x,self.y
    
    def recognition(self,img_digit):
        img_digit = cv2.cvtColor(img_digit,cv2.COLOR_BGR2HSV_FULL)
        img_digit = cv2.inRange(img_digit,(0,0,150),(255,50,255))
        scores = [0,0,0,0,0,0,0,0,0,0]
        img_rav = img_digit.ravel()
        if np.count_nonzero(img_rav)>10:
            for d in range(0,10):
                scores[d] = np.sum(np.abs(np.subtract(img_rav,self.digits[d])))/80
            return np.argmin(scores)
        else:
            return 0

    def load_digits(self):
        zero = np.array([   [  0,   0, 255, 255, 255,   0,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0, 255,   0,   0,   0, 255,   0,   0],
                            [  0, 255,   0,   0,   0, 255, 255,   0],
                            [  0, 255,   0,   0,   0, 255, 255,   0],
                            [  0, 255,   0,   0,   0, 255, 255,   0],
                            [  0, 255,   0,   0,   0, 255, 255,   0],
                            [  0, 255,   0,   0,   0, 255, 255,   0],
                            [  0, 255, 255,   0,   0, 255,   0,   0],
                            [  0,   0, 255, 255, 255, 255,   0,   0]]).ravel()

        one = np.array([[  0,   0,   0, 255,   0,   0,   0,  0],
                        [  0, 255, 255, 255, 255,   0,   0,  0],
                        [  0,   0,   0, 255, 255,   0,   0,  0],
                        [  0,   0,   0, 255, 255,   0,   0,  0],
                        [  0,   0,   0, 255, 255,   0,   0,  0],
                        [  0,   0,   0, 255, 255,   0,   0,  0],
                        [  0,   0,   0, 255, 255,   0,   0,  0],
                        [  0,   0,   0, 255, 255,   0,   0,  0],
                        [  0,   0,   0, 255, 255,   0,   0,  0],
                        [  0, 255, 255, 255, 255, 255, 255,  0]]).ravel()
        
        two = np.array([[  0,   0, 255, 255,   0,   0,   0,   0],
                        [  0, 255, 255, 255, 255, 255,   0,   0],
                        [  0,   0,   0,   0,   0, 255,   0,   0],
                        [  0,   0,   0,   0,   0, 255,   0,   0],
                        [  0,   0,   0,   0, 255, 255,   0,   0],
                        [  0,   0,   0,   0, 255,   0,   0,   0],
                        [  0,   0,   0, 255, 255,   0,   0,   0],
                        [  0,   0, 255, 255,   0,   0,   0,   0],
                        [  0, 255, 255,   0,   0,   0,   0,   0],
                        [  0, 255, 255, 255, 255, 255,   0,   0]]).ravel()
        
        three = np.array([  [  0,   0, 255, 255,   0,   0,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0,   0,   0,   0,   0, 255,   0,   0],
                            [  0,   0,   0,   0,   0, 255,   0,   0],
                            [  0,   0,   0, 255, 255, 255,   0,   0],
                            [  0,   0,   0, 255, 255, 255,   0,   0],
                            [  0,   0,   0,   0,   0, 255,   0,   0],
                            [  0,   0,   0,   0,   0, 255, 255,   0],
                            [  0,   0,   0,   0,   0, 255,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0]]).ravel()

        four = np.array([   [  0,   0,   0,   0, 255,   0,   0,   0],
                            [  0,   0,   0, 255, 255, 255,   0,   0],
                            [  0,   0,   0, 255, 255, 255,   0,   0],
                            [  0,   0, 255, 255, 255, 255,   0,   0],
                            [  0,   0, 255,   0, 255, 255,   0,   0],
                            [  0, 255,   0,   0, 255, 255,   0,   0],
                            [  0, 255,   0,   0, 255, 255,   0,   0],
                            [255, 255, 255, 255, 255, 255, 255,   0],
                            [  0,   0,   0,   0, 255, 255,   0,   0],
                            [  0,   0,   0,   0, 255, 255,   0,   0]]).ravel()

        five = np.array([   [  0,   0,   0,   0, 255, 255,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0, 255, 255,   0,   0,   0,   0,   0],
                            [  0, 255, 255,   0,   0,   0,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0,   0,   0,   0, 255, 255,   0,   0],
                            [  0,   0,   0,   0,   0, 255, 255,   0],
                            [  0,   0,   0,   0,   0, 255, 255,   0],
                            [  0,   0,   0,   0, 255, 255,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0]]).ravel()

        six = np.array([[  0,   0,   0, 255, 255, 255,   0,   0],
                        [  0,   0, 255, 255, 255, 255,   0,   0],
                        [  0, 255, 255,   0,   0,   0,   0,   0],
                        [  0, 255,   0,   0,   0,   0,   0,   0],
                        [  0, 255, 255, 255, 255, 255,   0,   0],
                        [  0, 255, 255,   0,   0, 255, 255,   0],
                        [  0, 255,   0,   0,   0, 255, 255,   0],
                        [  0, 255,   0,   0,   0, 255, 255,   0],
                        [  0, 255, 255,   0,   0, 255, 255,   0],
                        [  0,   0, 255, 255, 255, 255,   0,   0]]).ravel()

        seven = np.array([  [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0,   0,   0,   0,   0, 255,   0,   0],
                            [  0,   0,   0,   0, 255, 255,   0,   0],
                            [  0,   0,   0,   0, 255,   0,   0,   0],
                            [  0,   0,   0,   0, 255,   0,   0,   0],
                            [  0,   0,   0, 255, 255,   0,   0,   0],
                            [  0,   0,   0, 255,   0,   0,   0,   0],
                            [  0,   0,   0, 255,   0,   0,   0,   0],
                            [  0,   0, 255, 255,   0,   0,   0,   0]]).ravel()

        eight = np.array([  [  0,   0, 255, 255, 255,   0,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0, 255,   0,   0,   0, 255,   0,   0],
                            [  0, 255,   0,   0,   0, 255,   0,   0],
                            [  0,   0, 255, 255, 255, 255,   0,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0],
                            [  0, 255,   0,   0,   0, 255, 255,   0],
                            [  0, 255,   0,   0,   0, 255, 255,   0],
                            [  0, 255, 255,   0,   0, 255, 255,   0],
                            [  0, 255, 255, 255, 255, 255,   0,   0]]).ravel()

        nine =np.array([[  0,   0, 255, 255, 255,   0,   0,   0],
                        [  0, 255, 255, 255, 255, 255,   0,   0],
                        [  0, 255,   0,   0,   0, 255,   0,   0],
                        [  0, 255,   0,   0,   0, 255, 255,   0],
                        [  0, 255,   0,   0,   0, 255, 255,   0],
                        [  0, 255, 255,   0, 255, 255, 255,   0],
                        [  0,   0, 255, 255, 255, 255, 255,   0],
                        [  0,   0,   0,   0,   0, 255,   0,   0],
                        [  0,   0,   0,   0, 255, 255,   0,   0],
                        [  0, 255, 255, 255, 255,   0,   0,   0]]).ravel()

        return [zero,one,two,three,four,five,six,seven,eight,nine]