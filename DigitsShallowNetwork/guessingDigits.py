import numpy as np
import neuralNetwork as nnet
import cv2
import matplotlib.pyplot as plt
clicked = 0
endDraw = False
def mouse(event, x, y, flags, param):
    global clicked
    global endDraw
    if event == cv2.EVENT_LBUTTONDOWN and not clicked:
        clicked = 1
    elif event == cv2.EVENT_LBUTTONUP and clicked:
        clicked = 0
    if event == cv2.EVENT_MOUSEMOVE and clicked:
        cv2.circle(img, (x, y), 10, 1, -1)
    if event == cv2.EVENT_RBUTTONDBLCLK:
        endDraw = True

voting = nnet.Ensemble(9)
voting.load("9Ensemble")
img = np.zeros((280, 280))
outputImg = np.zeros((280, 280))
cv2.namedWindow("image")
cv2.namedWindow("output")
cv2.setMouseCallback("image", mouse)

while True:
    if np.any(img):
        output = cv2.resize(img, (img.shape[0] // 10, img.shape[1] // 10), interpolation=cv2.INTER_AREA)
        output.resize(784, 1)
        prediction = int(voting.vote([output])[0])
        cv2.putText(outputImg, str(prediction), (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, 255, 10)

    resized = cv2.resize(img, (img.shape[0] // 10, img.shape[1] // 10), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(resized, img.shape, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('image', resized)
    cv2.imshow("output", outputImg)
    outputImg = np.zeros((280, 280))

    if endDraw:
        cv2.destroyAllWindows()
        quit()
        break
        
    key = cv2.waitKey(1)
    if key == ord('a'):
        outputImg = np.zeros((280, 280))
        img = np.zeros((280, 280))
