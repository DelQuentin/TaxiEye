from matplotlib import pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

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

def segmentationv2(img,cam_set,sc,win):
    # print("\n Segmentation v2 Debug")
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

    first = True

    # ===== Process =====
    run = True
    while run:
        # print("New Line")
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
                # print(next_point)
                # print(3*winHalfSize[0],3*winHalfSize[1])
                if next_point[1] > 3*winHalfSize[1] and next_point[0] > 3*winHalfSize[0] and next_point[0] < 3*winHalfSize[0] + scale*real_w: # If point allows a window in the image
                    # print(next_point," is considered")
                    # Sliding Window Generation
                    sw = img[ next_point[1]-winHalfSize[1]:next_point[1]+winHalfSize[1] , next_point[0]-winHalfSize[0]:next_point[0]+winHalfSize[0] ]
                    seg_image = cv2.rectangle(seg_image,(next_point[0]-winHalfSize[0],next_point[1]-winHalfSize[1]),(next_point[0]+winHalfSize[0],next_point[1]+winHalfSize[1]),(150,150,150),1)
                    # Extract Lines
                    sw_full,dbg_extr = extractLines(sw,False)
                    # if first:
                    #     cv2.imshow("Extraction debug",dbg_extr)
                    #     cv2.waitKey(0)
                    #     first = False
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
        # plt.figure()
        # plt.plot(kernel_x,kernel_y)
        # plt.xlabel("Pixels")
        # plt.ylabel("Generated Line Kernel Density")
        # plt.show()
        # ===== From Kernel to Dots =====
        dtl_clusters = clusterize2(kernel_x,kernel_y,max(kernel_y)*0.5)
        dtl_dots = clusters2dots(dtl_clusters,0)
        # ===== Assign Point to Line =====
        # print("Lines : ",lines)
        # print("Lines to Process: ",line_to_process)
        # print("Dots : ",dtl_dots)
        dtl_matrix = np.zeros((len(line_to_process),len(dtl_dots)),np.uint)
        lines_dots = []
        for d in line_to_process:
            lines_dots.append([])
        # Matrix generation
        #  Line best match
        for l in range(len(line_to_process)):
            l_best_d = None
            min_d = 2*winHalfSize[0]
            for d in range(len(dtl_dots)):
                d_dtl = dtl_dots[d] - line_to_process[l][1][0]
                if abs(d_dtl) <= abs(min_d):
                    l_best_d = d
                    min_d = d_dtl
            if l_best_d != None:
                dtl_matrix[l][l_best_d] += 2
        #  Dot best match
        for d in range(len(dtl_dots)):
            d_best_l = None
            min_l = 2*winHalfSize[0]
            for l in range(len(line_to_process)):
                l_dtl = dtl_dots[d] - line_to_process[l][0][0]
                if abs(l_dtl) <= abs(min_l):
                    d_best_l = l
                    min_l = l_dtl
            if d_best_l != None:
                dtl_matrix[d_best_l][d] += 1
        #  Print Matrix
        # print(''.join(['\t'+str(d) for d in range(len(dtl_dots))]))
        # for l in range(len(line_to_process)):
        #     print(''.join([str(l)]+['\t'+str(dtl_matrix[l][d]) for d in range(len(dtl_dots))]))
        #  Assignation
        for d in range(len(dtl_dots)):
            l_assign = None
            l_score = 0
            for l in range(len(line_to_process)):
                if dtl_matrix[l][d] > l_score:
                    l_score = dtl_matrix[l][d]
                    l_assign = l
            if l_assign != None:
                lines_dots[l_assign].append(dtl_dots[d])
        # print("Lines Dots : ",lines_dots)
            
        # ========== Line Processing ==========
        for lp in range(len(lines_dots)):
            # Find next active line
            ls = 0
            for lss in range(len(lines)):
                if line_to_process[lp][0] == lines[lss][-1]:
                    ls = lss
                    break
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
    
    nb_points = 0
    for l in range(len(lines)):
        nb_points += len(lines[l])
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
        if l != 0 and lines[l][-1] not in joints and len(lines[l])<7:
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
    return final_lines,seg_image,nb_points

def seg_dbscan(img,cam_set,sc):
    # print("\n Segmentation v2 Debug")
    scale = sc # pixels per meter in the top to bottom image

    # ========== HOMOGRAPHIC TRANSFORM ==========
    # Extract lower part of image below horizon
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

    # Resize to real scale
    real_w = int(2*latFar)
    real_d = cam_set['depth']
    img = cv2.resize(transformed_img,[scale*real_w,scale*real_d])

    # Generate Image
    seg_image = np.zeros([scale*real_d,scale*real_w,3],dtype=np.uint8)
    seg_image = cv2.addWeighted(seg_image,1.0,img,0.3,0.0)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255)]

    # Extract Lines
    bin_img,dbg = extractLines(img,False)
    bin_sh = np.shape(bin_img)

    # Generate Segmentation Data Vector
    data_vec = []
    for i in range(bin_sh[0]):
        for j in range(bin_sh[1]):
            if bin_img[i][j] != 0:
                data_vec.append([i,j])
    clustering = DBSCAN(eps=10,min_samples=10).fit(data_vec)
    for k in range(len(data_vec)):
        seg_image = cv2.circle(seg_image,(data_vec[k][1],data_vec[k][0]),1,colors[clustering.labels_[k]%len(colors)])
    lines = []


    return lines, seg_image

# ==========================================================================================================================
# ===================================================== TOOL FUNCTIONS =====================================================
def extractLines(sw,debug):
    # cv2.imshow("sw",cv2.resize(sw,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    # Change color space
    sw_hsv = cv2.cvtColor(sw,cv2.COLOR_RGB2HSV_FULL)
    # cv2.imshow("hsv",cv2.resize(sw_hsv,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    # Filter color
    sw_bin = cv2.inRange(sw_hsv,(120,100,75),(155,255,255))
    # cv2.imshow("bin",cv2.resize(sw_bin,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    cv2.dilate(sw_bin,None,sw_bin,iterations=1)
    # cv2.imshow("bin_dil",cv2.resize(sw_bin,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    # Edge Detection
    sw_edges = cv2.Canny(sw,400,450)
    # cv2.imshow("edges",cv2.resize(sw_edges,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    # Combine and enhance
    sw_full = cv2.bitwise_or(sw_bin,sw_edges)
    # cv2.imshow("full",cv2.resize(sw_full,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    cv2.erode(sw_full,None,sw_full,iterations=1)
    # cv2.imshow("erode",cv2.resize(sw_full,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    cv2.dilate(sw_full,None,sw_full,iterations=1)
    # cv2.imshow("dilate",cv2.resize(sw_full,(25*np.shape(sw)[1],25*np.shape(sw)[0]),interpolation=cv2.INTER_AREA))
    # cv2.waitKey(0)
    # Return extracted lines mask
    dbg = []
    if debug:
        dbg = cv2.vconcat([sw,sw_hsv,cv2.cvtColor(sw_bin,cv2.COLOR_GRAY2BGR),cv2.cvtColor(sw_edges,cv2.COLOR_GRAY2BGR),cv2.cvtColor(sw_full,cv2.COLOR_GRAY2BGR)])
        dbg = cv2.resize(dbg,(5*np.shape(dbg)[1],5*np.shape(dbg)[0]),interpolation=cv2.INTER_AREA)
    return sw_full,dbg

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