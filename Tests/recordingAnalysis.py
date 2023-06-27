import cv2, time
import numpy as np

vidcap = cv2.VideoCapture('../Recordings/Taxi1End.mp4')
success,image = vidcap.read()
count = 0
width = np.shape(image)[1]
height = np.shape(image)[0]
while success:
  
    # ======== Image Treatment ========

    # Convert Color to HSV
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Taxiway Lines Extraction
    taxilines = cv2.inRange(hsv,np.array([10,100,100]),np.array([50,255,255]))

    # Find CenterLine
    xl = 0
    xr = 0
    foundL = False
    foundR = False
    while foundR == False and xl<width and xr<width:
        if foundL == False:
            if taxilines[height-1][xl] != 0 :
                foundL = True
                xr = xl + 1
            else:
                xl += 1
        else:
            if taxilines[height-1][xr] == 0 :
                foundR = True
            else:
                xr += 1

    # Computer Vision
    treated = np.zeros([height,width], dtype="uint8")
    treated = cv2.bitwise_and(image, image, mask=taxilines)
    if foundL == True:
        cv2.circle(treated, [(xr+xl)//2,height-1], 30, (0,0,255), 60)


    # ============ Display ============
    # Resize
    vision = cv2.resize(treated,[800,450])
    image = cv2.resize(image,[800,450])
    # Display
    cv2.imshow("Computer Vision", vision)
    cv2.imshow("Camera", image)

    # ====== Extract Next Frame =======
    success,image = vidcap.read()
    cv2.waitKey(1)
    time.sleep(1/120)