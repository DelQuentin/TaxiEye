# =============== Libraries Import ===============
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import cv2, sys, os, json
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ================= Files Import =================
from segmentation_func import segmentationv2,seg_dbscan
from matching_func import matching_with_hdg_sort_or_matrix
from Map import Map

# ===== PARAMETERS ===== (Adjust accodingly to system testing)
ds_path = sys.argv[1]
path_raw = ds_path + "raw/"
path_gt_ht = ds_path + "gt_ht/"
dataset_info_file = ds_path+"dataset.json"
seg_method = sys.argv[2]
eval_method = sys.argv[3]
# Load Dataset Information
with open(dataset_info_file, 'r') as f:
    ds_info = json.load(f)
cam = ds_info["cam"]

scale = 10
win = [2.5,1]
colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255)]
files = os.listdir(path_raw)
nb_files = len(files)
t = 0

X_true = []
X_pred = []

Acc = []
Pre = []
Rec = []
F1s = []
IoU = []
Point_m = []

for file in files:
    t += 1
    print("\n",t,"/",nb_files," : ",file)

    img = cv2.imread(path_raw+file)
    img_gt_ht = cv2.imread(path_gt_ht+file)
    h,w,c = np.shape(img)

    # ===== Generate Transformed Image Mask =====
    mask = 255*np.ones([h,w],dtype=np.uint8)
    src = np.array([[0,cam["top_limit"]],[w,cam["top_limit"]],[w,cam["bot_limit"]],[0,cam["bot_limit"]]],np.float32)
    # Data to caraterise field of view
    latFar = cam["far_half_width"]
    latNear = cam["near_half_width"]
    delta = w/2*latNear/latFar
    # Transformation
    dst = np.float32([[0,0], [w,0], [w/2+delta,h], [w/2-delta,h]]) 
    matrix = cv2.getPerspectiveTransform(src, dst)
    transformed_mask = cv2.warpPerspective(mask, matrix, (w,h))
    # resize to real scale
    real_w = int(2*latFar)
    real_d = cam['depth']
    mask = cv2.resize(transformed_mask,[scale*real_w,scale*real_d])
    # cv2.imshow("mask",mask)

    # ===== GENERATE DATA =====
    nb_points = 0
    if seg_method == 'SW':
        seg_data,dbg_image,nb_points = segmentationv2(img,cam,scale,win)
    elif seg_method == 'DBSCAN':
        seg_data,dbg_image = seg_dbscan(img,cam,scale)

    # ===== EVALUATION : LINES EXTRACTION FULL ===== Acc:0.9934122490911965  Pre:0.9311855366206824  Rec:0.648751282373315  F1:0.750377363323965  IoU:0.6219204661979955
    if eval_method == 'extraction':
        # Overall Taxi Lines Extraction Analysis (Using Shapes, to be redone using only point checks)
        seg_img = np.zeros(np.shape(dbg_image))
        # Add lines for label comparison on image
        for l in range(0,len(seg_data)):
            for p in range(1,len(seg_data[l])-1):
                cv2.line(seg_img,seg_data[l][p],seg_data[l][p+1],colors[l%len(colors)],8)
        seg_data_bin = cv2.inRange(cv2.cvtColor(np.float32(seg_img),cv2.COLOR_BGR2HSV),(0,1,1),(255,255,255))
        img_gt_ht_bin = cv2.inRange(cv2.cvtColor(np.float32(img_gt_ht),cv2.COLOR_BGR2HSV),(0,1,1),(255,255,255))
        i_TP = cv2.bitwise_and(mask, cv2.bitwise_and(seg_data_bin,img_gt_ht_bin))
        i_FP = cv2.bitwise_and(mask, cv2.bitwise_and(seg_data_bin,cv2.bitwise_xor(seg_data_bin,img_gt_ht_bin)))
        i_FN = cv2.bitwise_and(mask, cv2.bitwise_and(img_gt_ht_bin,cv2.bitwise_xor(seg_data_bin,img_gt_ht_bin)))
        i_TN = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_or(seg_data_bin,img_gt_ht_bin)))
        m_TP = cv2.countNonZero(i_TP)
        m_FP = cv2.countNonZero(i_FP)
        m_FN = cv2.countNonZero(i_FN)
        m_TN = cv2.countNonZero(i_TN)
        # === COMPUTE METRICS ===
        m_Acc = (m_TP+m_TN)/(m_TP+m_FP+m_TN+m_FN)
        m_Pre = m_TP/(m_TP+m_FP)
        m_Rec = m_TP/(m_TP+m_FN)
        m_F1 = 2*(m_Pre*m_Rec)/(m_Pre+m_Rec)
        m_IoU = m_TP/(m_TP+m_FP+m_FN)
        print("Accuracy:{}  Precision:{}  Recall:{}  F1:{}  IoU:{}".format(m_Acc,m_Pre,m_Rec,m_F1,m_IoU))
        Acc.append(m_Acc)
        Pre.append(m_Pre)
        Rec.append(m_Rec)
        F1s.append(m_F1)
        IoU.append(m_IoU)
        cv2.imshow("TP",i_TP)
        cv2.imshow("FN",i_FN)
        cv2.imshow("FP",i_FP)
        cv2.imshow("TN",i_TN)
        cv2.waitKey(0)
        # Visualise


    # ===== EVALUATION : LINE SEGMENTATION FULL ===== Acc:0.9984161371898215  Pre:0.5169393853703411  Rec:0.5432400652527388  F1:0.5162362699249281  IoU:0.4448578333541108
    elif eval_method == 'segmentation':
        # Match Classes
        line_color = []
        for line in seg_data:
            line_colors = {}
            for point in line[1:]:
                if point[1] <= len(img_gt_ht):
                    color_code = "-".join([str(col) for col in img_gt_ht[point[1],point[0]]])
                    if color_code in line_colors.keys():
                        line_colors[color_code] = line_colors[color_code] + 1
                    else:
                        line_colors[color_code] = 1
            if line_colors != {}:
                match_color = None
                match_nb = 0
                for co,nb in line_colors.items():
                    if nb > match_nb:
                        match_color = co
                        match_nb = nb
                line_color.append([int(col) for col in match_color.split('-')])
            else:
                line_color.append([255,0,0])

        # Line Reconstitution
        dbg_seg_img = np.zeros(np.shape(dbg_image))
        seg_img = np.zeros(np.shape(dbg_image))
        for l in range(0,len(seg_data)):
            for p in range(1,len(seg_data[l])-1):
                cv2.line(seg_img,seg_data[l][p],seg_data[l][p+1],line_color[l],8)
        # Compare GT and reconstitued and determine metrics per class
        # pixel_gt_color = []
        # pixel_seg_color = []
        m_classes = {}
        for i in range(np.shape(dbg_image)[0]):
            for j in range(np.shape(dbg_image)[1]):
                if mask[i,j] > 0:
                    color_code_seg = "-".join([str(int(col)) for col in seg_img[i,j]])
                    color_code_gt = "-".join([str(int(col)) for col in img_gt_ht[i,j]])
                    # pixel_seg_color.append(color_code_seg)
                    # pixel_gt_color.append(color_code_gt)
                    # Different Colors => Increment FP and FN
                    if color_code_seg != color_code_gt:
                        # SEG
                        if color_code_seg not in m_classes.keys():
                            m_classes[color_code_seg] = {}
                            m_classes[color_code_seg]["TP"] = 0
                            m_classes[color_code_seg]["TN"] = 0
                            m_classes[color_code_seg]["FN"] = 0
                            m_classes[color_code_seg]["FP"] = 1
                        else:
                            m_classes[color_code_seg]["FP"] += 1
                        # GT
                        if color_code_gt not in m_classes.keys():
                            m_classes[color_code_gt] = {}
                            m_classes[color_code_gt]["TP"] = 0
                            m_classes[color_code_gt]["TN"] = 0
                            m_classes[color_code_gt]["FN"] = 1
                            m_classes[color_code_gt]["FP"] = 0
                        else:
                            m_classes[color_code_gt]["FN"] += 1
                        dbg_seg_img[i,j] = [0,0,255]
                    # Same color => increment TP
                    else:
                        # SEG = GT
                        if color_code_seg not in m_classes.keys():
                            m_classes[color_code_seg] = {}
                            m_classes[color_code_seg]["TP"] = 1
                            m_classes[color_code_seg]["TN"] = 0
                            m_classes[color_code_seg]["FN"] = 0
                            m_classes[color_code_seg]["FP"] = 0
                        else:
                            m_classes[color_code_seg]["TP"] += 1
                        dbg_seg_img[i,j] = [0,255,0]
        
        # Compute Metrics
        m_macro_Acc = 0
        m_macro_Pre = 0
        m_macro_Rec = 0
        m_macro_F1 = 0
        m_macro_IoU = 0
        for c,e in m_classes.items():
            m_TP = m_classes[c]["TP"]
            m_FP = m_classes[c]["FP"]
            m_FN = m_classes[c]["FN"]
            m_TN = np.shape(dbg_image)[0]*np.shape(dbg_image)[1] - (m_classes[c]["TP"] + m_classes[c]["FP"] + m_classes[c]["FN"])
            # === COMPUTE METRICS ===
            m_Acc = (m_TP+m_TN)/(m_TP+m_FP+m_TN+m_FN)
            if (m_TP+m_FP) !=0 : 
                m_Pre = m_TP/(m_TP+m_FP)
            else: 
                m_Pre = 0
            m_Rec = m_TP/(m_TP+m_FN)
            if (m_Pre+m_Rec)!= 0:
                m_F1 = 2*(m_Pre*m_Rec)/(m_Pre+m_Rec)
            else:
                m_F1 = 0
            m_IoU = m_TP/(m_TP+m_FP+m_FN)
            # === Add to macro sum ===
            m_macro_Acc += m_Acc
            m_macro_Pre += m_Pre
            m_macro_Rec += m_Rec
            m_macro_F1 += m_F1
            m_macro_IoU += m_IoU
        # === Divide Macro
        m_macro_Acc = m_macro_Acc / len(m_classes.keys())
        m_macro_Pre = m_macro_Pre / len(m_classes.keys())
        m_macro_Rec = m_macro_Rec / len(m_classes.keys())
        m_macro_F1 = m_macro_F1 / len(m_classes.keys())
        m_macro_IoU = m_macro_IoU / len(m_classes.keys())
        print("Accuracy:{}  Precision:{}  Recall:{}  F1:{}  IoU:{}".format(m_macro_Acc,m_macro_Pre,m_macro_Rec,m_macro_F1,m_macro_IoU))
        Acc.append(m_Acc)
        Pre.append(m_Pre)
        Rec.append(m_Rec)
        F1s.append(m_F1)
        IoU.append(m_IoU)
        # Visualise
        cv2.imshow('SEG DEBUG',dbg_seg_img)
        # cm = confusion_matrix(pixel_gt_color,pixel_seg_color) + 1
        # sum = np.sum(np.sum(cm))
        # plot_confusion_matrix(100*cm/sum, cmap="viridis", norm_colormap=matplotlib.colors.LogNorm(), colorbar=True)
        # plt.show()
        cv2.waitKey(0)
    
    # ===== EVALUATION : POINT MATCHING ===== Acc:0.8729946253826794  Pre:0.45465057474345394  Rec:0.45444293396028773  F1:0.4540870667252219  IoU:0.44628845685844215
    if eval_method == 'matching':
        # ===== Generate Matching =====
        ds_map = Map(ds_info["map"])
        file_info = ds_info["data"][file]
        print(file_info)
        curr_line_info,next_cross_info = ds_map.situation_info(file_info["pos"],file_info["dir"])
        gt_labels,matching_score = matching_with_hdg_sort_or_matrix(seg_data,curr_line_info,next_cross_info,file_info["hdg"],False)
        print("GT Labels : ",gt_labels)

        # ========== POINT MATCHING ==========
        cor_points = 0
        mis_points = 0
        m_classes = {}
        for co,la in file_info["seg"].items():
            m_classes[la] = {}
            m_classes[la]["TP"] = 0
            m_classes[la]["TN"] = 0
            m_classes[la]["FN"] = 0
            m_classes[la]["FP"] = 0
        for l in range(len(seg_data)):
            line_label = gt_labels[l]
            if line_label == False: # If segmentation failed on the line, evaluation will count it as segmented as 'None' = no line
                line_label = 'None'
            for point in seg_data[l][1:]:
                if point[1] <= len(img_gt_ht):
                    color_code = "-".join([str(col) for col in img_gt_ht[point[1],point[0]]])
                    point_label = file_info["seg"][color_code]
                    if point_label == line_label:
                        dbg_image = cv2.putText(dbg_image,line_label,(point[0],point[1]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,255,50),2,2)
                        cor_points += 1
                        m_classes[line_label]["TP"] += 1
                    else:
                        dbg_image = cv2.putText(dbg_image,line_label+"/"+point_label,(point[0],point[1]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,50,255),2,2)
                        mis_points += 1
                        m_classes[line_label]["FP"] += 1
                        m_classes[point_label]["FN"] += 1
                    X_pred.append(point_label)
                    X_true.append(line_label)
        Point_m.append(cor_points/(cor_points+mis_points))
        # Compute Metrics
        m_macro_Acc = 0
        m_macro_Pre = 0
        m_macro_Rec = 0
        m_macro_F1 = 0
        m_macro_IoU = 0
        for c,e in m_classes.items():
            if c != "None":
                m_TP = m_classes[c]["TP"]
                m_FP = m_classes[c]["FP"]
                m_FN = m_classes[c]["FN"]
                m_TN = (cor_points+mis_points) - (m_classes[c]["TP"] + m_classes[c]["FP"] + m_classes[c]["FN"])
                print(c," - TP:",m_TP,"  FP:",m_FP,"  FN:",m_FN,"  TN:",m_TN)
                # === COMPUTE METRICS ===
                m_Acc = (m_TP+m_TN)/(m_TP+m_FP+m_TN+m_FN)
                if (m_TP+m_FP) !=0 : 
                    m_Pre = m_TP/(m_TP+m_FP)
                else: 
                    m_Pre = 0
                if (m_TP+m_FN) != 0:
                    m_Rec = m_TP/(m_TP+m_FN)
                else:
                    m_Rec = 0
                if (m_Pre+m_Rec)!= 0:
                    m_F1 = 2*(m_Pre*m_Rec)/(m_Pre+m_Rec)
                else:
                    m_F1 = 0
                if (m_TP+m_FP+m_FN) != 0:
                    m_IoU = m_TP/(m_TP+m_FP+m_FN)
                else:
                    m_IoU = 0
                # === Add to macro sum ===
                m_macro_Acc += m_Acc
                m_macro_Pre += m_Pre
                m_macro_Rec += m_Rec
                m_macro_F1 += m_F1
                m_macro_IoU += m_IoU
        # === Divide Macro
        m_macro_Acc = m_macro_Acc / (len(m_classes.keys())-1)
        m_macro_Pre = m_macro_Pre / (len(m_classes.keys())-1)
        m_macro_Rec = m_macro_Rec / (len(m_classes.keys())-1)
        m_macro_F1 = m_macro_F1 / (len(m_classes.keys())-1)
        m_macro_IoU = m_macro_IoU / (len(m_classes.keys())-1)
        print("Accuracy:{}  Precision:{}  Recall:{}  F1:{}  IoU:{}".format(m_macro_Acc,m_macro_Pre,m_macro_Rec,m_macro_F1,m_macro_IoU))
        Acc.append(m_Acc)
        Pre.append(m_Pre)
        Rec.append(m_Rec)
        F1s.append(m_F1)
        IoU.append(m_IoU)
        # Visualise
        cv2.imshow('DEBUG',dbg_image)
        cv2.waitKey(0)

    # ===== EVALUATION : LINE TO FOLLOW USING POINTS ===== Only Confusion Matrix
    if eval_method == 'follow_points': 
        # ===== Generate Matching =====
        ds_map = Map(ds_info["map"])
        file_info = ds_info["data"][file]
        path = file_info["follow"]
        curr_line_info,next_cross_info = ds_map.situation_info(file_info["pos"],file_info["dir"])
        labels,matching_score = matching_with_hdg_sort_or_matrix(seg_data,curr_line_info,next_cross_info,file_info["hdg"],False)
        # Generating the line to follow in the same way the navigator do, but with the position estimation process
        line_to_follow = []
        if len(seg_data) > 1 :
            for l in range(len(labels)):
                if labels[l] in path:
                    line_to_follow.extend(seg_data[l][1:])
        else:
            line_to_follow = seg_data[0][1:]

        # ========== POINT MATCHING ==========
        true = []
        pred = []
        for l in range(len(seg_data)):
            line_label = labels[l]
            if line_label == False: # If segmentation failed on the line, evaluation will count it as segmented as 'None' = no line
                line_label = 'None'
            for point in seg_data[l][1:]:
                if point[1] <= len(img_gt_ht):
                    # True
                    color_code = "-".join([str(col) for col in img_gt_ht[point[1],point[0]]])
                    point_label_gt = file_info["seg"][color_code]
                    if point_label_gt == "None":
                        X_true.append("None")
                    elif point_label_gt in path:
                        X_true.append("Line to Follow")
                    else:
                        X_true.append("Other lines")
                    # Prediction
                    if line_label == "None":
                        X_pred.append("None")
                    elif point in line_to_follow:
                        X_pred.append("Line to Follow")
                    else:
                        X_pred.append("Other lines")
        
    # ===== EVALUATION : LINE TO FOLLOW USING LINES ===== Only Confusion Matrix
    if eval_method == 'follow_lines': 
        # ===== Generate Matching =====
        ds_map = Map(ds_info["map"])
        file_info = ds_info["data"][file]
        path = file_info["follow"]
        curr_line_info,next_cross_info = ds_map.situation_info(file_info["pos"],file_info["dir"])
        labels,matching_score = matching_with_hdg_sort_or_matrix(seg_data,curr_line_info,next_cross_info,file_info["hdg"],False)
        # Generating the line to follow in the same way the navigator do, but with the position estimation process
        line_to_follow = []
        if len(seg_data) > 1 :
            for l in range(len(labels)):
                if labels[l] in path:
                    line_to_follow.extend(seg_data[l][1:])
        else:
            line_to_follow = seg_data[0][1:]

        # Mask of all deteted lines
        img_all_lines = np.zeros(np.shape(dbg_image))
        for l in range(0,len(seg_data)):
            for p in range(1,len(seg_data[l])-1):
                cv2.line(img_all_lines,seg_data[l][p],seg_data[l][p+1],(255,255,255),8)

        # Mask of all deteted lines to follow
        img_lines_to_follow = np.zeros(np.shape(dbg_image))
        for l in range(0,len(line_to_follow)-1):
                cv2.line(img_lines_to_follow,line_to_follow[l],line_to_follow[l+1],(255,255,255),8)

        for i in range(np.shape(dbg_image)[0]):
            for j in range(np.shape(dbg_image)[1]):
                if mask[i,j] > 0:
                    color_code_gt = "-".join([str(int(col)) for col in img_gt_ht[i,j]])
                    gt_label = file_info["seg"][color_code_gt]
                    color_code_all_lines = "-".join([str(int(col)) for col in img_all_lines[i,j]])
                    color_code_lines_to_follow = "-".join([str(int(col)) for col in img_lines_to_follow[i,j]])
                    # True
                    if gt_label in path:
                        X_true.append("Line to follow")
                    elif gt_label != 'None':
                        X_true.append("Other lines")
                    else:
                        X_true.append("None")

                    # Pred
                    if color_code_lines_to_follow != '0-0-0':
                        X_pred.append("Line to follow")
                    elif color_code_all_lines != '0-0-0':
                        X_pred.append("Other lines")
                    else:
                        X_pred.append("None")
# ===== RESULTS =====
print("\n == FINAL RESULTS ==")
# Extraction
if eval_method in ['segmentation','extraction']:
    print(eval_method," - Acc:{}  Pre:{}  Rec:{}  F1:{}  IoU:{}".format(np.mean(Acc),np.mean(Pre),np.mean(Rec),np.mean(F1s),np.mean(IoU)))

if eval_method == 'matching' : 
    plt.rcParams.update({'font.size': 8})
    cmd = ConfusionMatrixDisplay.from_predictions(X_true,X_pred,cmap='viridis',normalize='true', xticks_rotation='vertical', values_format='.2f')
    plt.show()

if eval_method == 'follow_points' :
    cmd = ConfusionMatrixDisplay.from_predictions(X_true,X_pred,cmap='viridis',normalize='true', xticks_rotation='vertical', values_format='.2f')
    plt.show()

if eval_method == 'follow_lines' :
    cmd = ConfusionMatrixDisplay.from_predictions(X_true,X_pred,cmap='viridis',normalize='true', xticks_rotation='vertical', values_format='.2f')
    plt.show()