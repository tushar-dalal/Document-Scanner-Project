import cv2
import numpy as np

imgHeight = 480
imgWidth = 640

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 0)


def preProcess(imag):
    imgGray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones([5, 5])
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThre = cv2.erode(imgDial, kernel, iterations=1)
    return imgThre


def getContours(imga):
    contours, hierarchy = cv2.findContours(imga, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = np.array([[0, 0]]*4)
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if area > maxArea & len(approx) == 4:
                biggest = approx
                maxArea = area
    #cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 10)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsnew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsnew[0] = myPoints[np.argmin(add)]
    myPointsnew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1);
    myPointsnew[1] = myPoints[np.argmin(diff)]
    myPointsnew[2] = myPoints[np.argmax(diff)]
    return myPointsnew


def getWarp(imag, biggest):
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(imag, matrix, (imgWidth, imgHeight))

    imgCropped = imgOutput[20: imgOutput.shape[0] - 20][20: imgOutput.shape[1]]
    imgCropped = cv2.resize(imgCropped, (480, 640))
    return imgCropped


while True:
    success, img = cap.read()
    cv2.resize(img, (imgWidth, imgHeight))
    imgContour = img.copy()
    imgThres = preProcess(img)
    big = getContours(imgThres)
    warped = getWarp(img, big)
    cv2.imshow("WebCam", warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
