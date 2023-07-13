import numpy as np
import cv2

def segmentation(img,cam_set,sc,win):
    scale = sc # pixels per meter in the top to bottom image

    # ========== HOMOGRAPHIC TRANSFORM ==========
    # extract lower part of image below horizon
    h,w,c = np.shape(img)
    src = np.array([[0,cam_set["top_limit"]],[w,cam_set["top_limit"]],[w,cam_set["bot_limit"]],[0,cam_set["bot_limit"]]],np.float32)

    # Data to caraterise field of view
    latFar = cam_set["far_half_width"]
    latNear = cam_set["near_half_width"]
    delta = w/2*latNear/latFar

    # Transformation
    dst = np.float32([[0,0], [w,0], [w/2+delta,h], [w/2-delta,h]]) 
    matrix = cv2.getPerspectiveTransform(src, dst)
    transformed_img = cv2.warpPerspective(img, matrix, (w,h))

    # resize to real scale
    real_w = int(2*latFar)
    real_d = cam_set['depth']
    img = cv2.resize(transformed_img,[scale*real_w,scale*real_d])

    # ========== SLIDING WINDOW ==========
    # ======== Init Params & Vars ========
    winHalfSize = [int(win[0]*scale),int(win[1]*scale)]
    lines = [[True,[int(scale*real_w//2),int(scale*real_d + winHalfSize[1])]]]
    seg_image = np.zeros([scale*real_d,scale*real_w,3],dtype=np.uint8)
    seg_image = cv2.addWeighted(seg_image,1.0,img,0.3,0.0)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255)]
    seg_image_alpha = np.zeros([scale*real_d,scale*real_w,3],dtype=np.uint8)

    # ===== Process =====
    run = True
    while run:
        # print("\n === SEG DEBUG ===")
        run = False
        line_to_process = []
        dots_to_process = []
        for line in lines:
            # If line is active
            if line[0] == True:
                # Line next Scan
                next_point = [line[-1][0],line[-1][1]-2*winHalfSize[1]]
                if len(line)>=3:
                    next_point[0] += (line[-1][0]-line[-2][0])

                if next_point[1] > 3*winHalfSize[1] and next_point[0] > winHalfSize[0] and next_point[0] < winHalfSize[0] + scale*real_w: # If point allows a window in the image
                    # Sliding Window Generation
                    sw = img[ next_point[1]-winHalfSize[1]:next_point[1]+winHalfSize[1] , next_point[0]-winHalfSize[0]:next_point[0]+winHalfSize[0] ]
                    # Extract Lines
                    sw_full = extractLines(sw)
                    # Create analyse horizontal data
                    hist = np.sum(sw_full/255,0).tolist()
                    # Clusters determination assuming 2 clusteres are always separated by a complete black column
                    clusters = clusterize(hist,0)
                    # Dots and Window
                    new_dots_X = clusters2dots(clusters,-winHalfSize[0])
                    # Store information for processing
                    exp = 0
                    if len(line)>=3:
                        exp = 2*(line[-1][0]-line[-2][0])
                    line_to_process.append([line[-1],next_point,exp])
                    for dot in new_dots_X:
                        dots_to_process.append([next_point[0]+dot,next_point[1]])
                    
                else:
                    line[0] == False
         
        # =============== Dots Further Treatment ===============
        if len(dots_to_process) == 0 or len(line_to_process) == 0:
            break
        # ========== Kernel Based Clusterisation ==========
        # ===== Kernel Initialisation =====
        kernel_x = []
        min_x = int(scale*real_w/2)
        max_x = int(scale*real_w/2)
        for line in line_to_process:
            min_x = min(min_x,line[1][0]-winHalfSize[0])
            max_x = max(max_x,line[1][0]+winHalfSize[0])
        kernel_x = range(min_x,max_x+1)
        kernel_y = [0]*len(kernel_x)
        # ===== Kernel Generation using gaussian distribution
        sig = 0.2*winHalfSize[0]
        gain = 1/(sig*np.sqrt(2*np.pi))
        for dot in dots_to_process:
            for k in range(len(kernel_x)):
                kernel_y[k] = max(kernel_y[k],gain*np.exp(-0.5*((kernel_x[k]-dot[0])/sig)**2))
        # ===== From Kernel to Dots =====
        dtl_clusters = clusterize2(kernel_x,kernel_y,max(kernel_y)*0.5)
        dtl_dots = clusters2dots(dtl_clusters,0)
        # ===== Assign Point to Line =====
        lines_dots = []
        for d in line_to_process:
            lines_dots.append([])
        for dot in dtl_dots:
            min_l = False
            min_d = winHalfSize[0]
            for l in range(len(line_to_process)):
                d = abs(dot-(line_to_process[l][1][0]))
                if d<min_d:
                    min_l = l
                    min_d = d 
            lines_dots[min_l].append(dot)
        # ========== Line Processing ==========
        ls = 0
        for lp in range(len(lines_dots)):
            # Find next active line
            while lines[ls][0] == False:
                ls+=1
            # Process
            if len(lines_dots[lp]) == 0:
                lines[ls][0] = False
            elif len(lines_dots[lp]) == 1:
                run = True
                lines[ls].append([lines_dots[lp][0],line_to_process[lp][1][1]])
            else:
                run = True
                lines[ls][0] = False
                for dot in lines_dots[lp]:
                    lines.append([True,line_to_process[lp][0],[dot,line_to_process[lp][1][1]]])
            # Go to next line
            ls+=1
    
    for l in range(0,len(lines)):
        cv2.circle(seg_image_alpha,lines[l][1],3,colors[l%len(colors)],4)
        for p in range(1,len(lines[l])-1):
            cv2.line(seg_image_alpha,lines[l][p],lines[l][p+1],colors[l%len(colors)],4)
            cv2.circle(seg_image_alpha,lines[l][p+1],3,colors[l%len(colors)],4)

    # ========== Line Post Processing ==========
    joints = []
    for line in lines:
        if line[1] not in joints:
            joints.append(line[1])
    l=0
    while l < len(lines):
        if lines[l][-1] not in joints and len(lines[l])<7:
            lines.pop(l)
        else:
            l+=1
    final_lines = []
    while len(lines)>0:
        line = lines.pop(0)
        runMerge = True
        while runMerge:
            runMerge = False
            lc = []
            for l in range(len(lines)):
                if lines[l][1] == line[-1]:
                    lc.append(l)
            if len(lc) == 1:
                runMerge = True
                nl = lines.pop(lc[0])
                for d in range(1,len(nl)):
                    line.append(nl[d])
        final_lines.append(line)
    # ===== Take Only Immidiate Crossing =====
    main_joint = final_lines[0][-1]
    l = 1
    while l<len(final_lines):
        if final_lines[l][1] != main_joint:
            final_lines.pop(l)
        else:
            l+=1

    # ===== Display Lines ===== 
    seg_image = cv2.addWeighted(seg_image,1.0,seg_image_alpha,0.3,0.0)
    for l in range(0,len(final_lines)):
        cv2.circle(seg_image,final_lines[l][1],1,colors[l%len(colors)],2)
        for p in range(1,len(final_lines[l])-1):
            cv2.line(seg_image,final_lines[l][p],final_lines[l][p+1],colors[l%len(colors)],2)
            cv2.circle(seg_image,final_lines[l][p+1],1,colors[l%len(colors)],2)

    # ===== End Of Process =====
    return final_lines,seg_image

# ==========================================================================================================================
# ===================================================== TOOL FUNCTIONS =====================================================
def extractLines(sw):
    # Change color space
    sw_hsv = cv2.cvtColor(sw,cv2.COLOR_RGB2HSV_FULL)
    # Filter color
    sw_bin = cv2.inRange(sw_hsv,(120,100,75),(155,255,255))
    # Edge Detection
    sw_edges = cv2.Canny(sw,400,450)
    # Combine and enhance
    sw_full = cv2.bitwise_or(sw_bin,sw_edges)
    cv2.erode(sw_full,None,sw_full,iterations=1)
    cv2.dilate(sw_full,None,sw_full,iterations=1)
    # Return extracted lines mask
    return sw_full

def clusterize(hist,th):
    clusters = []
    inCluster = False
    for i in range(len(hist)):
        if hist[i] <= th and inCluster==False:
            continue
        elif hist[i] <= th and inCluster==True:
            inCluster = False
        elif hist[i] > th and inCluster==False:
            inCluster = True
            clusters.append([])
            clusters[-1].append([i,hist[i]])
        else:
            clusters[-1].append([i,hist[i]]) 
    return clusters

def clusterize2(histX,histY,th):
    clusters = []
    inCluster = False
    for i in range(len(histY)):
        if histY[i] <= th and inCluster==False:
            continue
        elif histY[i] <= th and inCluster==True:
            inCluster = False
        elif histY[i] > th and inCluster==False:
            inCluster = True
            clusters.append([])
            clusters[-1].append([histX[i],histY[i]])
        else:
            clusters[-1].append([histX[i],histY[i]]) 
    return clusters

def clusters2dots(clusters,offset):
    dots = []
    for cl in clusters:
            mean = 0
            sum = 0
            for col in cl:
                mean += col[0]*col[1]
                sum += col[1]
            mean = mean / sum
            dots.append(int(mean+offset))
    return dots

def dist2(pt1,pt2):
    return np.sqrt((pt1[0]+pt2[0])**2+(pt1[1]+pt2[1])**2)