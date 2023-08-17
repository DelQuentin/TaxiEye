import cv2, sys, os
import numpy as np

SEGMENTATION = True
seg_labels = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255]]

# ===== Paths & Files =====
path_from = sys.argv[1]
path_to = sys.argv[2]
files = os.listdir(path_from)
nb_files = len(files)
t = 0

# ===== PARAMETERS ===== (Adjust accodingly to system testing)
cam_set = {
    "top_limit": 140,                                           # Assumed level of end of vision on the ground, just under vanishing line (in pixels)
    "bot_limit": 1050,                                          # Assumed level of begin of vision on the ground (just over the telemetry stripe in DCS) (in pixels)
    "depth": 60,                                                # Distance in meters between the aircraft and the end of vision on the ground (in meters)
    "near_half_width": 1.75,                                    # Distance between center of vision and lateral view limit at level of begin of vision on the ground (in meters)
    "far_half_width": 55.75,                                    # Distance between center of vision and lateral view limit at level of end of vision on the ground (in meters)
}
# cam_set = {
#     "top_limit": 120,
#         "bot_limit": 1050,
#         "depth": 53,
#         "near_half_width": 3.5,
#         "far_half_width": 21,
# }
scale = 10 # pixels per meter in the top to bottom image

# ========== HOMOGRAPHIC TRANSFORM FOR DATASET IMAGES ==========
for file in files:
    img = cv2.imread(path_from+file)
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

    # Post treatment to cancel the distaortion and restore the class/labels
    if SEGMENTATION:
        for i in range(scale*real_d):
            for j in range(scale*real_w):
                if img[i][j][0] != 0 or img[i][j][1] != 0 or img[i][j][2] != 0:
                    corres = [255]*len(seg_labels)
                    for k in range(len(seg_labels)):
                        corres[k] = np.sqrt((seg_labels[k][0]-img[i][j][0])**2+(seg_labels[k][1]-img[i][j][1])**2+(seg_labels[k][2]-img[i][j][2])**2)
                    label_idx = np.argmin(corres)
                    if corres[label_idx] <= 150:
                        img[i][j] = seg_labels[label_idx]
                    else:
                        img[i][j] = [0,0,0]

    # Display
    t += 1
    print(t,"/",nb_files," : ",file)
    cv2.imshow("Transformed Image",img)
    cv2.waitKey(1)

    # Export
    cv2.imwrite(path_to+file,img)