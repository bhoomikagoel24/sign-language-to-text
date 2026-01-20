import cv2
import numpy as np
import math
import time
import os
from cvzone.HandTrackingModule import HandDetector

# === SETTINGS ===
LABEL = "C"  # change this for each class you want to collect
imgSize = 300
offset = 20
folder = f"Data/{LABEL}"

if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

counter = 0

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Collected Image", imgWhite)

    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)

    if key == ord("s"):  
        counter += 1
        filename = f"{folder}/{LABEL}_{int(time.time())}.jpg"
        cv2.imwrite(filename, imgWhite)
        print(f"[{counter}] Saved ->", filename)

    if key == 27 or key == ord("q"):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
