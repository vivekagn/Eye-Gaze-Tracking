# Facial landmarks
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math

# Constants for controlling the Blink Frequency.
blinkThreshold = 0.3
blinkFrameThresh = 3
blinkCounter = 0
blinks = 0

def eyeRatio(eye):
    # Possibly replace euclidean() with manual calculation
    
    # Calculate average height of eye
    h1 = dist.euclidean(eye[1], eye[5])
    h2 = dist.euclidean(eye[2], eye[4])
    heightAverage = (h1 + h2) / 2.0

    # Calculate width of eye
    width = dist.euclidean(eye[0], eye[3])

    # Calculate the eye aspect ratio
    ratio = heightAverage / width

    return ratio

def eyeRegion(eye):
    xmax, ymax = eye[0]
    xmin, ymin = eye[0]
    for (x,y) in eye:
        if x > xmax:
            xmax = x
        if x < xmin:
            xmin = x
        if y > ymax:
            ymax = y
        if y < ymin:
            ymin = y
    
    return frame[ymin:ymax, xmin:xmax]

def get_iris_center(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img = cv2.medianBlur(img,5)
    #thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
    _,thresh = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    #kernel = np.ones((3,3),np.uint8)
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    #diff = thresh - closed
    thresh[thresh == 0] = 127
    thresh[thresh == 255] = 0
    thresh[thresh == 127] = 255
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) > 0:
        # Get center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
        
        cv2.drawContours(image, [c], 0, (0, 255, 0), 1)
    #for c in cnts:
    #    cv2.drawContours(image, [c], 0, (0, 255, 0), 1)
    
    cv2.imshow("Eye", image)
    cv2.imshow("Thresh", thresh)
    #cv2.imshow("Closed", closed)
    #cv2.imshow("Diff", diff)
    cv2.waitKey(1)


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Use Webcam
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()

    if ret == False:
        break

    # Used for FPS calculation
    startTime = cv2.getTickCount()

    # Load frame from webcam, resize and convert to grayscale
    #frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in frame
    faces = detector(gray, 1)

    # loop over the face detections
    for (i, face) in enumerate(faces):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # Get coordinates of 6 eye features
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # Calulate ratio of eyes
        leftEyeRatio = eyeRatio(leftEye)
        rightEyeRatio = eyeRatio(rightEye)
        
        # Get left and right eye
        lEye = eyeRegion(leftEye)
        rEye = eyeRegion(rightEye)
        
        # Get center of each eye
        get_iris_center(lEye)
        get_iris_center(rEye)
        
        # Calculate average eye aspect ratio
        averageEyeRatio = (leftEyeRatio + rightEyeRatio)/2.0


        leftCorner = leftEye[0]
        rightCorner = leftEye[3]

        angle = math.atan((leftCorner[1]-rightCorner[1]) / (leftCorner[0]-rightCorner[0])) 
        #print(angle)
    
        # Calculate blink score based on how closed eye is
        if averageEyeRatio < 0.32:
            score += (0.35 - averageEyeRatio)
        else:
            if score > 0.10:
                blinks += 1
            score = 0

#         if(averageEyeRatio < blinkThreshold):
#             blinkCounter += 1
#         else:
#             if blinkCounter >= blinkFrameThresh:
#                 blinks += 1

#             blinkCounter = 0

        # Get bounding box coordinates
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (255, 0, 0), 1)

        # show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, "Eye Ratio: {}".format(averageEyeRatio), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Calculate frames per second (fps)
    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - startTime))
    
    cv2.putText(frame, "FPS: {}".format(fps), (10, 265),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("WebCam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

# webcam is still in use if python dosent quit
quit()