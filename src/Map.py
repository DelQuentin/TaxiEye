import numpy as np
import cv2,json

class Map:
    def __init__(self,map_name):
        self.map_name = map_name
        file_name = ''.join(['src/Maps/',map_name,'.json'])
        map_data = {}
        with open(file_name, 'r') as f:
            map_data = json.load(f)
        self.lines = map_data["lines"]
        self.X = map_data["X"]
        self.Y = map_data["Y"]
        self.bounds = map_data["Boundaries"]
        self.display_scale = 1

    def cv2XY(self,x,y):
        return [self.X-x,self.Y-y]
    
    def map_image(self,path):
        # Determine Dimensions in original scale (meters)
        map_w = self.bounds[3]-self.bounds[2]
        map_h = self.bounds[1]-self.bounds[0]
        # Determine Display Scale
        self.display_scale = 1600/map_w
        if map_h*self.display_scale > 900:
            self.display_scale = 900/map_h
        # Create Image
        # map_img = 50*np.ones([int(self.display_scale*map_h),int(self.display_scale*map_w),3],dtype=np.uint8)
        map_img = cv2.imread(''.join(['src/Maps/Background_',self.map_name,'.png']))
        map_img = cv2.resize(map_img,[int(self.display_scale*map_w),int(self.display_scale*map_h)])
        # Draw Lines
        for name,dots in self.lines.items():
            x_s = int(self.display_scale*(self.bounds[3]-dots["Y_s"]))
            y_s = int(self.display_scale*(dots["X_s"]-self.bounds[0]))
            x_e = int(self.display_scale*(self.bounds[3]-dots["Y_e"]))
            y_e = int(self.display_scale*(dots["X_e"]-self.bounds[0]))
            if name in path:
                cv2.line(map_img,[x_s,y_s],[x_e,y_e],(0,0,255),1)
            else:
                cv2.line(map_img,[x_s,y_s],[x_e,y_e],(0,200,255),1)
        # Display
        return map_img
    
    def point_on_image(self,map,x,y,r,c,t):
        x_m = int(self.display_scale*(self.bounds[3]-y))
        y_m = int(self.display_scale*(x-self.bounds[0]))
        return cv2.circle(map,[x_m,y_m],r,c,t)

    def situation_info(self,curr_line,curr_dir):
        curr_line_info = []
        next_cross_info = []
        line = self.lines[curr_line]
        # Parking position
        if curr_dir == 0:
            if line["N_s"]=={}:
                curr_line_info = [curr_line,line["H_e"],1]
                for l,d in line["N_e"].items():
                    next_cross_info.append([l,self.line_heading(l,d)])
            else:
                curr_line_info = [curr_line,line["H_s"],-1]
                for l,d in line["N_s"].items():
                    next_cross_info.append([l,self.line_heading(l,d)])
        # Line direction 1 : as defined in JSON
        elif curr_dir == 1:
            curr_line_info = [curr_line,self.line_heading(curr_line,1),1]
            for l,d in line["N_e"].items():
                next_cross_info.append([l,self.line_heading(l,d)])
        # Line direction -1 : inverse of as defined in JSON
        elif curr_dir == -1:
            curr_line_info = [curr_line,self.line_heading(curr_line,-1),-1]
            for l,d in line["N_s"].items():
                next_cross_info.append([l,self.line_heading(l,d)])
        return curr_line_info,next_cross_info
    
    def line_heading(self,line,dir):
        if dir == -1:
            return np.mod(np.arctan2(self.lines[line]["Y_e"]-self.lines[line]["Y_s"],self.lines[line]["X_e"]-self.lines[line]["X_s"])*180/np.pi,360)
        elif dir == 1:
            return np.mod(np.arctan2(self.lines[line]["Y_s"]-self.lines[line]["Y_e"],self.lines[line]["X_s"]-self.lines[line]["X_e"])*180/np.pi,360)
        else:
            return False

if __name__ == "__main__":
    creech = Map('Creech')
    cv2.imshow("Map",creech.map_image([]))
    cv2.waitKey(0)