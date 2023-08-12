import numpy as np

def matching(vision_lines,curr_line_info,next_cross_info,debug):
    # ===== Variable Initialisation =====
    score = 0
    matrix = np.zeros([len(vision_lines)-1,len(next_cross_info)])
    labels = [False]*len(vision_lines)
    # Heading Correction
    if len(vision_lines[0])>2:
        th_i_pp = heading_img(vision_lines[0][-2],vision_lines[0][-1])
    else:
        th_i_pp = 0
    if debug: print("th_i_pp : ",th_i_pp)
    th_i_p = curr_line_info[1]
    if debug: print("th_i_p : ",th_i_p)
    th = [0]*len(vision_lines)
    for l in range(len(vision_lines)):
        th[l] = heading_img_corr(vision_lines[l][1],vision_lines[l][-1],th_i_p,th_i_pp)
    if debug: print("th",th)

    # ===== Matrix generation =====
    for vl in range(1,len(vision_lines)):
        for ml in range(len(next_cross_info)):
            matrix[vl-1][ml] = abs(heading_diff(th[vl],next_cross_info[ml][1]))
    # Display Matrix
    if debug:
        print("".join(["\t"]+["\t|"+l[0] for l in next_cross_info]))
        print("".join(["\t"]+["\t|"+str(np.round(l[1],2))+"\t" for l in next_cross_info]))
        for l in range(0,len(vision_lines)-1):
            print("".join([str(l),"\t",str(np.round(th[1+l],2))]+["\t|"+str(np.round(p,2))+"\t" for p in matrix[l]]))

    # ===== Matching =====
    labels[0] = curr_line_info[0]
    for l in range(1,len(vision_lines)):
        # Determine minimum
        min_flat = np.argmin(matrix)
        i = min_flat // len(next_cross_info)
        j = min_flat % len(next_cross_info)
        # Add distance square to score sum
        score += matrix[i][j]**2
        # Store min in labels
        labels[i+1] = next_cross_info[j][0]
        # Disable line and column in minimum search
        matrix[i,:] = 180
        matrix[:,j] = 180

    score = np.sqrt(score)
    return labels,score

def matching_with_hdg_matrix(vision_lines,curr_line_info,next_cross_info,hdg,debug):
    # ===== Variable Initialisation =====
    score = 0
    matrix = np.zeros([len(vision_lines)-1,len(next_cross_info)])
    labels = [False]*len(vision_lines)
    th = [0]*len(vision_lines)
    for l in range(len(vision_lines)):
        th[l] = heading_img_corr(vision_lines[l][1],vision_lines[l][-1],hdg,0)
    if debug: print("th",th)

    # ===== Matrix generation =====
    for vl in range(1,len(vision_lines)):
        for ml in range(len(next_cross_info)):
            matrix[vl-1][ml] = abs(heading_diff(th[vl],next_cross_info[ml][1]))
    # Display Matrix
    if debug:
        print("".join(["\t"]+["\t|"+l[0] for l in next_cross_info]))
        print("".join(["\t"]+["\t|"+str(np.round(l[1],2))+"\t" for l in next_cross_info]))
        for l in range(0,len(vision_lines)-1):
            print("".join([str(l),"\t",str(np.round(th[1+l],2))]+["\t|"+str(np.round(p,2))+"\t" for p in matrix[l]]))

    # ===== Matching =====
    labels[0] = curr_line_info[0]
    for l in range(1,min(len(vision_lines),1+len(next_cross_info))):
        # Determine minimum
        min_flat = np.argmin(matrix)
        i = min_flat // len(next_cross_info)
        j = min_flat % len(next_cross_info)
        # Add distance square to score sum
        score += matrix[i][j]**2
        # Store min in labels
        labels[i+1] = next_cross_info[j][0]
        # Disable line and column in minimum search
        matrix[i,:] = 180
        matrix[:,j] = 180

    score = np.sqrt(score)
    return labels,score

def matching_with_hdg_sort(vision_lines,curr_line_info,next_cross_info,hdg,debug):
    # ===== Variable Initialisation =====
    score = 0
    labels = [False]*len(vision_lines)
    labels[0] = curr_line_info[0]
    if len(vision_lines)>1:
        th = []
        for l in range(1,len(vision_lines)):
            hdg_l = heading_img_corr(vision_lines[l][1],vision_lines[l][-1],hdg,0)
            if hdg_l > 180:
                th.append(hdg_l-360)
            else:
                th.append(hdg_l)
        if debug: print("th",th)
        map_th = []
        for cl in next_cross_info:
            if cl[1] > 180:
                map_th.append(cl[1]-360)
            else:
                map_th.append(cl[1])
        if debug: print("map_th",map_th)
        
        # ===== Heading Sort =====
        vis_hdg_sort,vis_hdg_sort_idx = sort_plus_idx(th)
        if debug: print("Sorted : vis ",vis_hdg_sort," idx ",vis_hdg_sort_idx)
        map_hdg_sort,map_hdg_sort_idx = sort_plus_idx(map_th)
        if debug: print("Sorted : map ",map_hdg_sort," idx ",map_hdg_sort_idx)
    
        # ===== Matching =====
        if len(vis_hdg_sort) == len(map_hdg_sort):
            for k in range(len(vis_hdg_sort)):
                labels[vis_hdg_sort_idx[k]+1] = next_cross_info[map_hdg_sort_idx[k]][0]
    
    return labels,score

def matching_with_hdg_sort_or_matrix(vision_lines,curr_line_info,next_cross_info,hdg,debug):
    # ===== Variable Initialisation =====
    score = 0
    labels = [False]*len(vision_lines)
    labels[0] = curr_line_info[0]
    if len(vision_lines)>1:
        th = []
        for l in range(1,len(vision_lines)):
            hdg_l = heading_img(vision_lines[l][1],vision_lines[l][-1])
            if hdg_l > 180:
                th.append(hdg_l-360)
            else:
                th.append(hdg_l)
        if debug: print("th",th)
        map_th = []
        for cl in next_cross_info:
            hdg_ml = heading_diff(hdg,cl[1])
            if hdg_ml > 180:
                map_th.append(hdg_ml-360)
            else:
                map_th.append(hdg_ml)
        if debug: print("map_th",map_th)

        # ===== Matching =====
        if len(vision_lines)-1 == len(next_cross_info):
            # ===== Heading Sort =====
            if debug: print("Sort Matching")
            vis_hdg_sort,vis_hdg_sort_idx = sort_plus_idx(th)
            if debug: print("Sorted : vis ",vis_hdg_sort," idx ",vis_hdg_sort_idx)
            map_hdg_sort,map_hdg_sort_idx = sort_plus_idx(map_th)
            if debug: print("Sorted : map ",map_hdg_sort," idx ",map_hdg_sort_idx)
            # ===== Sort Matching =====
            for k in range(len(vis_hdg_sort)):
                labels[vis_hdg_sort_idx[k]+1] = next_cross_info[map_hdg_sort_idx[k]][0]
        else:
            if debug: print("Matrix Matching")
            matrix = np.zeros([len(vision_lines)-1,len(next_cross_info)])
            # ===== Matrix generation =====
            for vl in range(1,len(vision_lines)):
                for ml in range(len(next_cross_info)):
                    matrix[vl-1][ml] = abs(heading_diff(th[vl-1],next_cross_info[ml][1]))
            # Display Matrix
            if debug:
                print("".join(["\t"]+["\t|"+l[0] for l in next_cross_info]))
                print("".join(["\t"]+["\t|"+str(np.round(l[1],2))+"\t" for l in next_cross_info]))
                for l in range(0,len(vision_lines)-1):
                    print("".join([str(l),"\t",str(np.round(th[l],2))]+["\t|"+str(np.round(p,2))+"\t" for p in matrix[l]]))
            # ===== Matrix Matching =====
            for l in range(1,min(len(vision_lines),1+len(next_cross_info))):
                # Determine minimum
                min_flat = np.argmin(matrix)
                i = min_flat // len(next_cross_info)
                j = min_flat % len(next_cross_info)
                # Add distance square to score sum
                score += matrix[i][j]**2
                # Store min in labels
                labels[i+1] = next_cross_info[j][0]
                # Disable line and column in minimum search
                matrix[i,:] = 180
                matrix[:,j] = 180
    
    return labels, score

# ==========================================================================================================================
# ===================================================== TOOL FUNCTIONS =====================================================
def heading_img(p1,p2):
    return np.mod(-np.arctan2(p1[0]-p2[0],p1[1]-p2[1])*180/np.pi,360)

def heading_img_corr(p1,p2,th_i_p,th_i_pp):
    return np.mod((-np.arctan2(p1[0]-p2[0],p1[1]-p2[1])*180/np.pi)+th_i_p-th_i_pp,360)

def heading_diff(h1,h2):
    diff = (h1 - 360) if h1 > 180 else h1
    h2 = h2 - diff
    return h2 if h2 < 180 else h2 - 360

def sort_plus_idx(li: list):
    sorted = li.copy()
    sorted.sort()
    idx_list = []
    for elt in sorted:
        idx_list.append(li.index(elt))
    return sorted,idx_list