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

def matching_with_hdg(vision_lines,curr_line_info,next_cross_info,hdg,debug):
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