from Map import Map
from matching_func import matching,matching_with_hdg_matrix,matching_with_hdg_sort,matching_with_hdg_sort_or_matrix
import matplotlib.pyplot as plt
import numpy as np

class Navigator:

    def __init__(self, init_pos: str, init_dir: int, path: list, map: Map):
        self.pos = init_pos
        self.dir = init_dir
        self.path = path
        self.map = map
        self.matching = []
        self.dist_deriv = []
        self.dist = []
        self.pos_flag = False
    
    def run_no_avionics(self,seg_data):

        # Run matching of current line
        curr_line_info,next_cross_info = self.map.situation_info(self.pos,self.dir)
        labels,matching_score = matching(seg_data,curr_line_info,next_cross_info,False)
        self.matching.append(matching_score)
        line_to_follow = []
        if len(seg_data) > 1 :
            for l in range(len(labels)):
                if labels[l] in self.path:
                    line_to_follow.extend(seg_data[l][1:])
        else:
            line_to_follow = seg_data[0][1:]

        # Distance Data Analytics
        self.dist.append(seg_data[0][-1][1])
        if len(self.dist) > 2:
            self.dist_deriv.append(self.dist[-1]-self.dist[-2])
        else : self.dist_deriv.append(0)

        # Determine if crossing has been passed
        cross_passed = False
        if self.dist_deriv[-1] < 0 and self.dist[-2] >= 600:
            switch_n = 0
            for rec in range(1,101):
                if self.dist_deriv[-(1+rec)] < 0:
                    switch_n -= 1
                else:
                    switch_n += 1
            if switch_n > 95:
                cross_passed = True
            else:
                cross_passed = False
        
        # Switch to next crossing with choice
        if cross_passed:
            pos_idx = self.path.index(self.pos)
            for idx in range(pos_idx + 1, len(self.path)):
                nb = 0
                new_dir = 0
                if self.path[idx-1] in self.map.lines[self.path[idx]]["N_s"]:
                    nb = len(self.map.lines[self.path[idx]]["N_e"])
                    new_dir = 1
                elif self.path[idx-1] in self.map.lines[self.path[idx]]["N_e"]:
                    nb = len(self.map.lines[self.path[idx]]["N_s"])
                    new_dir = -1
                # if more than 1 line, then stop to focus on this crossing next time
                if nb > 1:
                    self.pos = self.path[idx]
                    self.dir = new_dir
                    break
                # or end of path
                elif idx == len(self.path)-1:
                    self.pos = self.path[idx]
                    self.dir = new_dir
        return labels,line_to_follow
    
    def run_with_avionics(self,seg_data,gps,hdg):

        # ===== Gather data for navigation =====
        # Take current situation information
        end_path = False
        curr_line_info,next_cross_info = self.map.situation_info(self.pos,self.dir)
        # Determine corrected position
        corr_pos,end = self.project_on_map_line(self.pos,gps[0],gps[1]) 
        # Distance Data Analytics
        self.dist.append(seg_data[0][-1][1])
        if len(self.dist) > 2:
            self.dist_deriv.append(self.dist[-1]-self.dist[-2])
        else : self.dist_deriv.append(0)

        switch_line = False
        # ===== Position Flag for line switch =====
        if self.pos_flag == False:
            if curr_line_info[2] == end:
                self.pos_flag = True
                # If flag raised while in last path line, it means the end of the path is reached
                if self.pos == self.path[-1]:
                    end_path = True
        # ===== Vision Flag for line switch =====
        # Switch line if position flag is raised and no crossing is in sight
        if self.pos_flag and (self.dist_deriv[-1] < -100 or len(seg_data) == 1 or len(next_cross_info)<=1):
            switch_line = True
        
        if switch_line:
            # Updtae current line
            self.pos = self.path[min(self.path.index(self.pos)+1,len(self.path))]
            # Update current direction
            new_idx = self.path.index(self.pos)
            if self.path[new_idx-1] in self.map.lines[self.path[new_idx]]["N_s"]:
                self.dir = 1
            else : 
                self.dir = -1
            # Reset position line switch flag
            self.pos_flag = False

        # Generate matching and line to follow
        labels,matching_score = matching_with_hdg_sort_or_matrix(seg_data,curr_line_info,next_cross_info,hdg,True)
        self.matching.append(matching_score)
        line_to_follow = []
        if len(seg_data) > 1 :
            for l in range(len(labels)):
                if labels[l] in self.path:
                    line_to_follow.extend(seg_data[l][1:])
        else:
            line_to_follow = seg_data[0][1:]

        return labels,line_to_follow,corr_pos,end_path
        

    # =========== TOOL FUNCTIONS ==========

    def project_on_map_line(self,line,x,y):
        p1 = [self.map.lines[line]['X_s'],self.map.lines[line]['Y_s']]
        p2 = [self.map.lines[line]['X_e'],self.map.lines[line]['Y_e']]
        nb_iter = 7
        for iter in range(nb_iter):
            d1 = np.sqrt((p1[0]-x)**2+(p1[1]-y)**2)
            d2 = np.sqrt((p2[0]-x)**2+(p2[1]-y)**2)
            mid = [ (p1[0]+p2[0])/2 , (p1[1]+p2[1])/2 ]
            if d1 < d2:
                p2 = mid
            else:
                p1 = mid
        # Indicate if close to end of line, and which end   
        end = 0
        if p1 == [self.map.lines[line]['X_s'],self.map.lines[line]['Y_s']] or p2 == [self.map.lines[line]['X_s'],self.map.lines[line]['Y_s']]:
            end = -1
        elif p1 == [self.map.lines[line]['X_e'],self.map.lines[line]['Y_e']] or p2 == [self.map.lines[line]['X_e'],self.map.lines[line]['Y_e']]:
            end = 1
        return mid,end
            
    def get_data(self) -> list:
        return self.matching, self.dist_deriv, self.dist