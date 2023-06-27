import numpy as np
import cv2,json

class Map:
    def __init__(self,map_name,route):
        self.map_name = map_name
        file_name = ''.join(['src/Maps/',map_name,'.json'])
        map_data = {}
        with open(file_name, 'r') as f:
            map_data = json.load(f)
        self.lines = map_data["lines"]
        self.X = map_data["X"]
        self.Y = map_data["Y"]
        self.bounds = map_data["Boundaries"]
        self.am = self.makeAdjacencyMatrix(self.lines)
        self.route = route
    
    def makeAdjacencyMatrix(self,lines):
        return []

    def cv2XY(self,x,y):
        return [self.X-x,self.Y-y]
    
    def display(self):
        # Determine Dimensions in original scale (meters)
        map_w = self.bounds[3]-self.bounds[2]
        map_h = self.bounds[1]-self.bounds[0]
        print(map_w,map_h)
        # Determine Display Scale
        display_scale = 1600/map_w
        if map_h*display_scale > 900:
            display_scale = 900/map_h
        print(display_scale)
        # Create Image
        # map_img = 50*np.ones([int(display_scale*map_h),int(display_scale*map_w),3],dtype=np.uint8)
        map_img = cv2.imread(''.join(['src/Maps/Background_',self.map_name,'.png']))
        map_img = cv2.resize(map_img,[int(display_scale*map_w),int(display_scale*map_h)])
        print(np.shape(map_img))
        # Draw Lines
        for name,dots in self.lines.items():
            color = (0,200,255)
            if name in self.route:
                color = (0,0,255)
            x_s = int(display_scale*(self.bounds[3]-dots["Y_s"]))
            y_s = int(display_scale*(dots["X_s"]-self.bounds[0]))
            x_e = int(display_scale*(self.bounds[3]-dots["Y_e"]))
            y_e = int(display_scale*(dots["X_e"]-self.bounds[0]))
            cv2.line(map_img,[x_s,y_s],[x_e,y_e],color,1)
        # Display
        cv2.imshow("Airport Map",map_img)

if __name__ == "__main__":
    creech = Map('Creech',["P2","A2_P2_3","A3","A3_P3_1","A4","A4_F1_2","F1"])
    creech.display()
    cv2.waitKey(0)
