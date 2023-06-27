import numpy as np
import cv2,datetime

def segmentation(img,viewLimit,debug):
    if debug: print("\n ===== Segmentation =====")
    if debug: print(str(datetime.datetime.now())," - Process Start")
    scale = 10

    # ========== HOMOGRAPHIC TRANSFORM ==========
    # extract lower part of image below horizon
    h,w,c = np.shape(img)
    src = np.array([[0,viewLimit],[w,viewLimit],[w,h],[0,h]],np.float32)

    # Data to caraterise field of view
    latFar = 49
    latNear = 1.67
    delta = w/2*latNear/latFar

    # Transformation
    dst = np.float32([[0,0], [w,0], [w/2+delta,h], [w/2-delta,h]]) 
    matrix = cv2.getPerspectiveTransform(src, dst)
    transformed_img = cv2.warpPerspective(img, matrix, (w,h))

    # resize to real scale
    real_w = 98
    real_d = 60
    img = cv2.resize(transformed_img,[scale*real_w,scale*real_d])
    if debug: print(str(datetime.datetime.now())," - Transformation Successful")

    # ========== SLIDING WINDOW ==========
    # ===== Init Params & Vars =====
    winHalfSize = [scale*3,scale*1]
    lines = [[True,[int(scale*real_w//2),int(scale*real_d + winHalfSize[1])]]]
    if debug:
        seg_debug = np.zeros([scale*real_d,scale*real_w,3],dtype=np.uint8)
        seg_debug = cv2.addWeighted(seg_debug,1.0,img,0.3,0.0)
    mrg_th = 1*winHalfSize[0]

    # ===== Process =====
    run = True
    while run:
        run = False
        for line in lines:
            # If line is active
            if line[0] == True:
                # Line next Scan
                next_point = [line[-1][0],line[-1][1]-2*winHalfSize[1]]
                if len(line)>=3:
                    next_point[0] += (line[-1][0]-line[-2][0])

                if next_point[1] > winHalfSize[1] and next_point[0] > winHalfSize[0] and next_point[0] < winHalfSize[0] + scale*real_w: # If point allows a window in the image
                    # Sliding Window Generation
                    sw = img[ next_point[1]-winHalfSize[1]:next_point[1]+winHalfSize[1] , next_point[0]-winHalfSize[0]:next_point[0]+winHalfSize[0] ]
                    if debug:
                        cv2.rectangle(seg_debug,(next_point[0]-winHalfSize[0], next_point[1]-winHalfSize[1]),(next_point[0]+winHalfSize[0], next_point[1]+winHalfSize[1]),(127,127,127),1)
                    # Extract Lines
                    sw_full = extractLines(sw)
                    # Create analyse horizontal data
                    hist = np.sum(sw_full/255,0).tolist()
                    # Clusters determination assuming 2 clusteres are always separated by a complete black column
                    clusters = hist2clusters(hist)
                    # Dots and Window
                    new_dots_X = clusters2dots(clusters,winHalfSize)
                    if len(new_dots_X)!=0:
                        run = True
                        # Only one dot means line follows
                        if len(new_dots_X)==1:
                            line.append([next_point[0]+new_dots_X[0],next_point[1]])
                        # Multiple Dots : find best and determine if others are noise, if noise, clear and continue lie, if not, stop line and create new ones
                        else:
                            filtered_dots = []
                            # Find best point for line continuation
                            line_next_dot = 0
                            min_d = winHalfSize[0]
                            for dot in new_dots_X:
                                exp = 0
                                if len(line)>=3:
                                    exp = 2*(line[-1][0]-line[-2][0])
                                if abs(dot-exp) < min_d:
                                    min_d = abs(dot)
                                    line_next_dot = dot
                            # Look at other dots to determine which are noise
                            for dot in new_dots_X:
                                add = True
                                if dot != line_next_dot:
                                    for other_line in lines:
                                        if other_line[-1][1] == next_point[1] and abs(other_line[-1][0] == next_point[0]+dot)<mrg_th: #To close to another line point
                                            add = False
                                if add:
                                    filtered_dots.append(dot)
                            # Mode than 1 dot, start new lines
                            if len(filtered_dots)>1:
                                line[0] = False
                                for dot in filtered_dots:
                                    lines.append([True,line[-1],[next_point[0]+dot,next_point[1]]])
                            # only one viable dot, continue line on it
                            else:
                                line.append([next_point[0]+line_next_dot,next_point[1]]) # Aligned with the dot : UPGRADE POSSIBLE : Estimate deviation of next point for better window frame  
                    # no new dots
                    else:
                        line[0] = False

        # Lines Post-Treatment - Remove Duplicate Lines
        lenl = len(lines)
        line1 = 1
        while line1 < lenl:
            line2=line1+1
            while line2 < lenl:
                if abs(lines[line1][-1][0]-lines[line2][-1][0])<0.1*winHalfSize[0]:
                    lines.pop(line2)
                    lenl = len(lines)
                line2+=1
            line1+=1

    # ===== Display Lines ===== 
    if debug:
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255)]
        for l in range(0,len(lines)):
            cv2.circle(seg_debug,lines[l][1],2,colors[l%len(colors)],2)
            for p in range(1,len(lines[l])-1):
                cv2.line(seg_debug,lines[l][p],lines[l][p+1],colors[l%len(colors)],2)
                cv2.circle(seg_debug,lines[l][p+1],2,colors[l%len(colors)],2)

    # ===== End Of Process =====
    if debug: print(str(datetime.datetime.now())," - Process Successful")
    if debug: cv2.imshow('Segmentation Debug',seg_debug)
    return lines

# ==========================================================================================================================
# ===================================================== TOOL FUNCTIONS =====================================================
def extractLines(sw):
    # Change color space
    sw_hsv = cv2.cvtColor(sw,cv2.COLOR_RGB2HSV_FULL)
    cv2.imshow("HSV",cv2.resize(sw_hsv,[600,200],interpolation=cv2.INTER_AREA))
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

def hist2clusters(hist):
    clusters = []
    inCluster = False
    for i in range(len(hist)):
        if hist[i] == 0 and inCluster==False:
            continue
        elif hist[i] == 0 and inCluster==True:
            inCluster = False
        elif hist[i] != 0 and inCluster==False:
            inCluster = True
            clusters.append([])
            clusters[-1].append([i,hist[i]])
        else:
            clusters[-1].append([i,hist[i]]) 
    return clusters

def clusters2dots(clusters,winHalfSize):
    dots = []
    for cl in clusters:
            mean = 0
            sum = 0
            for col in cl:
                mean += col[0]*col[1]
                sum += col[1]
            mean = mean / sum
            dots.append(int(mean-winHalfSize[0]))
    return dots

def dist2(pt1,pt2):
    return np.sqrt((pt1[0]+pt2[0])**2+(pt1[1]+pt2[1])**2)