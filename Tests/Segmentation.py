import numpy as np
import matplotlib.pyplot as plt
import cv2,datetime,colorsys
from sklearn.cluster import AgglomerativeClustering

def segmentation(img,viewLimit,debug):
    if debug: print("\n ===== Segmentation =====")
    if debug: print(str(datetime.datetime.now())," - Process Start")
    scale = 10

    # ===== HOMOGRAPHIC TRANSFORM =====
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

    # ===== SLIDING WINDOW =====
    # Init
    winHalfSize = [scale*1,scale*3]
    winX_init = int(scale*real_w//2)
    winY_init = int(scale*real_d - winHalfSize[0])
    windows = [[winY_init,winX_init]]
    if debug:
        seg_debug = np.zeros([scale*real_d,scale*real_w,3],dtype=np.uint8)
        seg_debug = cv2.addWeighted(seg_debug,1.0,img,0.3,0.0)

    # ===== Process =====
    dots = []
    run = True
    while run:
        new_windows=[]
        new_dots = []
        line_dots = []
        for window in windows:
            if debug:
                cv2.rectangle(seg_debug,(window[1]-winHalfSize[1], window[0]-winHalfSize[0]),(window[1]+winHalfSize[1], window[0]+winHalfSize[0]),(255,0,0),2)

            # Create Sliding Window
            sw = img[ window[0]-winHalfSize[0]:window[0]+winHalfSize[0] , window[1]-winHalfSize[1]:window[1]+winHalfSize[1] ]
            
            # Extract Lines
            sw_full = extractLines(sw)

            # Create analyse horizontal data
            hist = np.sum(sw_full/255,0).tolist()

            # Clusters determination assuming 2 clusteres are always separated by a complete black column
            clusters = hist2clusters(hist)

            # Dots and Window
            new_dots_X = clusters2dots(clusters,winHalfSize)
            for dot_X in new_dots_X:
                new_dots.append([window[1]-winHalfSize[1],window[1]+dot_X,window[1]+winHalfSize[1],window[0]])
        
        # Line Dots Duplicate Erasing
        line_dots = []
        for ndot in new_dots:
            add = True
            for odot in line_dots:
                if ndot[1]>odot[0] and ndot[1]<odot[2]:
                    if ndot[0] > odot[0] and abs(ndot[1]-odot[1])<odot[2]-ndot[0]:
                        add = False
                        break
                    if ndot[0] < odot[0] and abs(ndot[1]-odot[1])<ndot[2]-odot[0]:
                        add = False
                        break
            if add:
                line_dots.append(ndot)

        # Next Windows
        for dot in line_dots:
            
            new_windows.append([dot[3]-2*winHalfSize[0],dot[1]])
        windows = new_windows

        # End of process
        dots.append(line_dots)
        for window in windows:
            if window[0] < winHalfSize[0] or window[0] > scale*real_d-winHalfSize[0] or window[1] < winHalfSize[1] or window[1] > scale*real_w-winHalfSize[1]:
                run = False
                break
        if run == True and len(windows)==0:
            run = False
    # ==========
    
    # Dots Clusters and Lines
    lines = []
    for dot_line in dots:
        # First Line Case
        if len(lines) == 0:
            for dot in dot_line:
                lines.append([[dot[1],dot[3]]])
        # Nominal Case
        else:
            on_line = [False]*len(lines)
            no_line = []
            for dot in dot_line:
                line = 0
                min_d = 2*scale*(real_d+real_w)
                for l in range(len(lines)):
                    d = dist2([dot[1],dot[3]],lines[l][-1])
                    if d <= min_d:
                        min_d = d
                        line = l
                # Line free
                if on_line[line] == False:
                    on_line[line] = [[dot[1],dot[3]],min_d,l]
                # Line not free but better match
                elif on_line[line][1] > min_d:
                    # Remove the one in place
                    no_line.append(on_line[line[0]])
                    # Replace it
                    on_line[line] = [[dot[1],dot[3]],min_d,l]
                # Line not free and not best match
                else:
                    no_line.append([dot[1],dot[3]])
            for l in on_line:
                if l != False:
                    lines[l[2]].append(l[0])
            for l in no_line:
                lines.append([l])
    print(lines)
                

    # Display Lines
    if debug:
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255)]
        for l in range(len(lines)):
            cv2.circle(seg_debug,lines[l][0],2,colors[l%len(colors)],2)
            for p in range(len(lines[l])-1):
                cv2.line(seg_debug,lines[l][p],lines[l][p+1],colors[l%len(colors)],2)
                cv2.circle(seg_debug,lines[l][p+1],2,colors[l%len(colors)],2)

    if debug: print(str(datetime.datetime.now())," - Process Successful")
    if debug: cv2.imshow('Segmentation Debug',seg_debug)

    if len(lines)>1:
            cv2.waitKey(0)

    return lines



# ==========================================================================================================================
# ===================================================== TOOL FUNCTIONS =====================================================
def extractLines(sw):
    # Change color space
    sw_hsv = cv2.cvtColor(sw,cv2.COLOR_RGB2HSV_FULL)
    # Filter color
    sw_bin = cv2.inRange(sw_hsv,(120,70,70),(155,255,255))
    cv2.dilate(sw_bin,None,sw_bin,iterations=1)
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
            dots.append(int(mean-winHalfSize[1]))
    return dots

def dist2(pt1,pt2):
    return np.sqrt((pt1[0]+pt2[0])**2+(pt1[1]+pt2[1])**2)

# ==========================================================================================================================
# ======================================================= DEPRECATED =======================================================
# # Show
# if debug:
#     sw_lines = np.zeros((2*winHalfSize[0],2*winHalfSize[1],3),dtype=np.uint8)
#     for line in lines:
#         sw_lines = cv2.line(sw_lines,(line+winHalfSize[1],2*winHalfSize[0]),(line+winHalfSize[1],0),[0,0,255],1)
#     # Display Line
#     plt.figure(figsize=(12,4))
#     plt.subplot(231)
#     plt.imshow(cv2.flip(cv2.cvtColor(sw,cv2.COLOR_BGR2RGB),0),origin="lower")
#     plt.title("Sliding Window")
#     plt.axis('off')
#     plt.subplot(232)
#     plt.imshow(cv2.flip(cv2.cvtColor(sw_hsv,cv2.COLOR_BGR2RGB),0),origin="lower")
#     plt.title("HSV Color Space")
#     plt.axis('off')
#     plt.subplot(233)
#     plt.imshow(cv2.flip(cv2.cvtColor(sw_bin,cv2.COLOR_BGR2RGB),0),origin="lower")
#     plt.title("HSV Filtered")
#     plt.axis('off')
#     plt.subplot(234)
#     plt.imshow(cv2.flip(cv2.cvtColor(sw_edges,cv2.COLOR_BGR2RGB),0),origin="lower")
#     plt.title("Edge Filtered")
#     plt.axis('off')
#     plt.subplot(235)
#     plt.imshow(cv2.flip(cv2.cvtColor(sw_full,cv2.COLOR_BGR2RGB),0),origin="lower")
#     plt.title("Combined Filtering")
#     plt.axis('off')
#     plt.subplot(236)
#     plt.imshow(cv2.flip(cv2.cvtColor(sw_lines,cv2.COLOR_BGR2RGB),0),origin="lower")
#     plt.title("Segmented Lines")
#     plt.axis('off')
#     plt.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99, hspace=0.01, wspace=0.1)
#     plt.show()