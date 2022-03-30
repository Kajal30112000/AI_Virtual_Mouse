import cv2
import numpy as np
import os
import time
import HandTrackingModule2 as htm


brushThickness=10
eraserThickness=50

folderPath="header"
mylist=os.listdir(folderPath)
print(mylist)
overlayList=[]
for imPath in mylist:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header=overlayList[0]
drawColor=(255,0,255)
wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector=htm.handDetector(detectionCon=0.95)
xp,yp=0,0
imgCanvas=np.zeros((720,1280,3),np.uint8)

while True:
    # 1.import img

    success, img = cap.read()
    img=cv2.flip(img,1)

    # 2. find hand landmarks
    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)

    if len(lmlist)!=0:

        #print(lmlist)

        x1,y1=lmlist[8][1:]   #tip of index and middle fingers
        x2, y2 = lmlist[12][1:]

    # 3. check which fingers are up         0 for open , 1 for closed finger

        fingers = detector.fingersUp()
        #print(fingers)
        # 4. if selection mode--2 finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            print("Selection Mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor =(0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            cv2.rectangle(imgCanvas, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. drawing Mode--index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),10,drawColor,cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp,yp=x1,y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    img[0:125,0:1280]=header    # setting the header image
   # img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)


    cv2.imshow("Image",img)
    cv2.imshow("Image canvas", imgCanvas)
    cv2.imshow("Image inverse", imgInv)
    cv2.waitKey(1)