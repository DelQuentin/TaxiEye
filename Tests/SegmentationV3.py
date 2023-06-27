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
    mrg_th = int(0.3*winHalfSize[0]) # Dots merging threshold

    # ===== Process =====
    run = True
    while run:
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
                    # Store information for processing
                    exp = 0
                    if len(line)>=3:
                        exp = 2*(line[-1][0]-line[-2][0])
                    line_to_process.append([line[-1],next_point[0]+exp])
                    for dot in new_dots_X:
                        dots_to_process.append([[next_point[0]+dot,next_point[1]],line[-1]])
        
        # ===== Dots Further Treatment =====
        print("\n === DEBUG ===")
        print("lines:",line_to_process)
        print("dots:",dots_to_process)
        # Proces Points : Match point to best line in process
        lines_dots = []
        for d in line_to_process:
            lines_dots.append([])
        print(lines_dots)
        for dot in dots_to_process:
            min_l = False
            min_d = winHalfSize[0]
            for l in range(len(line_to_process)):
                d = abs(dot[0][0]-line_to_process[l][1])
                if d<min_d:
                    min_l = l
                    min_d = d 
            lines_dots[min_l].append(dot)
        print("lines dots:",lines_dots)
        # Line Processing
        ls = 0
        for lp in range(len(lines_dots)):
            # Find next active line
            while lines[ls][0] == False:
                ls+=1
            # Sort dots to keep best match as first in list and remmove duplicates 
            line_dots = [[]]
            for dot in lines_dots[lp]:
                if line_dots[0] == []:
                    line_dots[0] = dot
                else:
                    # Replace first if better
                    if d==0 and abs(dot[0][0]-line_to_process[lp][1]) < abs(line_dots[d][0][0]-line_to_process[lp][1]):
                        line_dots.insert(0,dot)
                    # Search for (close) duplicate
                    add = True
                    for d in range(len(line_dots)):
                        if abs(dot[0][0]-line_dots[d][0][0]) < mrg_th:
                            add = False
                            break
                    if add:
                        line_dots.append(dot)
            print("Line {}({}) : {}".format(lp,ls,line_dots))
            # Process
            if line_dots[0] == []:
                lines[ls][0] = False
            elif len(line_dots) == 1:
                run = True
                lines[ls].append(line_dots[0][0])
            else:
                run = True
                lines[ls][0] = False
                for dot in line_dots:
                    lines.append([True,dot[1],dot[0]])
            # Go to next line
            ls+=1
            
        print("Line Debug")
        for line in lines:
            print(line)

        # ===== Display Lines ===== 
        if debug:
            colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255)]
            for l in range(0,len(lines)):
                cv2.circle(seg_debug,lines[l][1],2,colors[l%len(colors)],2)
                for p in range(1,len(lines[l])-1):
                    cv2.line(seg_debug,lines[l][p],lines[l][p+1],colors[l%len(colors)],2)
                    cv2.circle(seg_debug,lines[l][p+1],2,colors[l%len(colors)],2)
            cv2.imshow('Segmentation Debug',seg_debug)

    # ===== End Of Process =====
    if debug: print(str(datetime.datetime.now())," - Process Successful")
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