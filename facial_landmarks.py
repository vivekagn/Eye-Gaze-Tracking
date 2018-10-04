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
	frame = imutils.resize(frame, width=500)
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

		# Calculate average eye aspect ratio
		averageEyeRatio = (leftEyeRatio + rightEyeRatio)/2.0


		leftCorner = leftEye[0]
		rightCorner = leftEye[3]

		angle = math.atan((leftCorner[1]-rightCorner[1]) / (leftCorner[0]-rightCorner[0])) 
		print(angle)

		if(averageEyeRatio < blinkThreshold):
			blinkCounter += 1
		else:
			if blinkCounter >= blinkFrameThresh:
				blinks += 1

			blinkCounter = 0

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
