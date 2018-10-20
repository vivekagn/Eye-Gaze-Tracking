# Facial landmarks
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math
import sys

# Constants for controlling the Blink Frequency.
blinkThreshold = 0.3
blinkFrameThresh = 3
blinkCounter = 0
blinks = 0
score = 0

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

def gammaCorrection(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    gamma = 1.0 / gamma
    lookupTable = np.array([((i / 255.0) ** gamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
 
    gammaCorrected = cv2.LUT(image, lookupTable)
    
    return gammaCorrected

def get_iris_center(image, eye, gamma=1.0):
    # Increased size to width of 100 for better results
    image = imutils.resize(image, width=100)
    # Convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gamma correction
    img = gammaCorrection(img, gamma)

    # Return to caller to adjust gamma in future call
    meanAfterGamma = cv2.mean(img)[0]

    img = cv2.GaussianBlur(img, (5,5), 0)

    # Normalise image after gamma correction
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("normalised", img)

    # Binarise image
    _,thresh = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    
    # Closing
    # kernel = np.ones((3,3),np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Opening
    # kernel = np.ones((5,5),np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Invert binarisation
    thresh[thresh == 0] = 127
    thresh[thresh == 255] = 0
    thresh[thresh == 127] = 255
    
    cv2.imshow("{} Eye threshold".format(eye), thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    # Check if contour is found
    if len(cnts) == 0:
        return meanAfterGamma
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) > 0:
        # Get center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        arcLength = cv2.arcLength(c, True)
        contourArea = cv2.contourArea(c)
        Pi = 3.14159
        # Calculate approximate radius based on arc length of contour
        # L = 2  * Pi * r
        radiusL = arcLength / (2 * Pi)
        radiusLSquared = radiusL * radiusL
        # Calculate approximate radius based on area of contour
        # A = Pi * r^2
        radiusASquared = contourArea / Pi
        # Use the ratio of radii from arc length and area to determine circularity of contour
        circularity = radiusLSquared / radiusASquared

        # print("radius ratio = {}. x = {}, y = {}. Area = {}, Length = {}".format(radiusRatio, cX, cY, contourArea, arcLength))

        # reject contour if circularity is out of acceptable range
        if circularity > 1.4 or circularity < 0.75:
            return meanAfterGamma

        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        
        cv2.drawContours(image, [c], 0, (0, 255, 0), 1)
    
    cv2.imshow("{} Eye".format(eye), image)

    cv2.waitKey(1)

    return meanAfterGamma


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Use sample video
if len(sys.argv[:]) == 2:
    filename = sys.argv[1]
    camera = cv2.VideoCapture(filename)

# Use Webcam
else:
    camera = cv2.VideoCapture(0)

# Initial gamma
gamma = 1.0
# Initial mean intensity of eye region
mean = 0

while camera.isOpened():
    ret, frame = camera.read()

    if ret == False:
        break

    # Used for FPS calculation
    startTime = cv2.getTickCount()

    # Load frame from webcam, resize and convert to grayscale
    frame = imutils.resize(frame, width=600)
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
        # if leftEyeRatio > 0.3:
            # get_iris_center(lEye, "Left")
        if rightEyeRatio > 0.25:
            # Adjust gamma to suit brightness
            if mean > 35:
                gamma -= 0.02
            elif mean < 30:
                gamma += 0.02
            mean = get_iris_center(rEye, "Right", gamma=gamma)
        
        # Calculate average eye aspect ratio
        averageEyeRatio = (leftEyeRatio + rightEyeRatio)/2.0


        leftCorner = leftEye[0]
        rightCorner = leftEye[3]

        angle = math.atan((leftCorner[1]-rightCorner[1]) / (leftCorner[0]-rightCorner[0]))
    
        # Calculate blink score based on eye aspect ratio
        if averageEyeRatio < 0.32:
            score += (0.35 - averageEyeRatio)
        else:
            if score > 0.10:
                blinks += 1
            score = 0

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