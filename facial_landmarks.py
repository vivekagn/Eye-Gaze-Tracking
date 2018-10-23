# Facial landmarks
from scipy.spatial import distance as dist
from imutils import face_utils
from Helper import *
import numpy as np
import imutils
import dlib
import cv2
import math
import random
import sys

class EyeGaze:
	def __init__(self):
		# Constants for controlling the Blink Frequency.
		self.blinkThreshold = 0.30
		# Minimum number of frames for intentional blink or wink to be registered
		self.blinkFrameThresh = 4
		self.blinkFrameCounter = 0
		self.leftEyeWinkCounter = 0
		self.rightEyeWinkCounter = 0
		# Total number of blinks
		self.blinks = 0
		self.blinkScore = 0

		self.helper = Helper()
		# Face detector
		self.detector = dlib.get_frontal_face_detector()
		# Facial landmark predictor is used to find facial features such as eyes and nose
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		self.colour = (0,225,0)

		# Initial gamma
		self.gammaLeft = 1.0
		self.gammaRight = 1.0
		# Initial mean intensity of eye region
		self.meanLeft = 0
		self.meanRight = 0

		# Upper and lower bounds of mean
		self.meanUpper = 80
		self.meanLower = 70

		self.screenWidth = 1400
		self.controlArea = np.ones((100, self.screenWidth, 3), np.uint8) * 255

	def start(self, arg=""):
		# Use sample video
		if len(arg) != 0:
			camera = cv2.VideoCapture(arg)

		else:
			camera = cv2.VideoCapture(0)

		while camera.isOpened():
			# Get frame from webcam
			ret, frame = camera.read()

			if ret == False:
				break

			# Display control window
			cv2.imshow("Control Area", self.controlArea)

			# Coordiates of pupil for left and right eyes
			xL = yL = xR = yR = 0

			# Used for FPS calculation
			startTime = cv2.getTickCount()

			# Resize and convert to grayscale
			frame = imutils.resize(frame, width=600)
			frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Detect faces in frame
			faces = self.detector(frameGray, 1)

			# loop over the face detections
			for (i, face) in enumerate(faces):
				# Obtain facial landmarks
				FaceLandmarks = self.predictor(frameGray, face)
				FaceLandmarks = face_utils.shape_to_np(FaceLandmarks)

				# Get first and last index of coordinates corresponding to each eye
				(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
				(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

				(lEyebrowStart, lEyebrowEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
				(rEyebrowStart, rEyebrowEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

				# Get coordinates of eyebrows
				leftEyebrow = FaceLandmarks[lEyebrowStart:lEyebrowEnd]
				rightEyebrow = FaceLandmarks[rEyebrowStart:rEyebrowEnd]

				# Get coordinates of 6 eye features
				leftEye = FaceLandmarks[lStart:lEnd]
				rightEye = FaceLandmarks[rStart:rEnd]

				eyebrowToEyeDistanceRatio = dist.euclidean(leftEye[4], leftEyebrow[2]) / dist.euclidean(rightEye[5], rightEyebrow[2])
				# print("Dist = {}".format(eyebrowRatio))

				# Calulate aspect ratio of eyes
				leftEyeRatio = self.helper.eyeRatio(leftEye)
				rightEyeRatio = self.helper.eyeRatio(rightEye)

				# Get left and right eye
				lEye = self.helper.eyeRegion(leftEye, frame)
				rEye = self.helper.eyeRegion(rightEye, frame)

				# Get center of each eye
				if leftEyeRatio > 0.25:
					self.helper.get_iris_center(lEye, "Left")
					# Adjust gamma to suit brightness
					if self.meanLeft > self.meanUpper:
						self.gammaLeft += 0.02
					elif self.meanLeft < self.meanLower:
						self.gammaLeft -= 0.02
					meanL, (xL, yL) = self.helper.get_iris_center(lEye, "Left", gamma=self.gammaLeft)
					self.meanLeft += meanL
					self.meanLeft = self.meanLeft / 2

				if rightEyeRatio > 0.25:
					# Adjust gamma to suit brightness
					if self.meanRight > self.meanUpper:
						self.gammaRight += 0.02
					elif self.meanRight < self.meanLower:
						self.gammaRight -= 0.02
					meanR, (xR, yR) = self.helper.get_iris_center(rEye, "Right", gamma=self.gammaRight)
					self.meanRight += meanR
					self.meanRight = self.meanRight / 2

				# print("{} {}".format(mean,gamma))
				# print("xL = {}, xR = {}".format(int(xL), int(xR)))

				# Pupil located for both eyes
				if xL > 0 and xR > 0:
					x = int((xL + xR) * 50)
					# print("Left x: {}, y: {} Right x: {}, y: {}".format(int(xL * 100), int(yL * 100), int(xR * 100), int(yR * 100)))
				# Pupil located only for left eye
				elif xL > 0:
					x = int(xL * 100)
				
				# Pupil located only for right eye
				elif xR > 0:
					x = int(xR * 100)
				
				# Pupil cannot be located for either eye
				else:
					x = 0

				# Pupil located for at least one eye
				# Update control area
				if x != 0:
					# Clear control area
					self.controlArea[:, :, :] = 255
					# 40, 60 are locations in eye region corresponding to looking left and right on screen
					# Will need to add in a calibration so these aren't hard coded in
					if x < 40: x = 40
					if x > 60: x = 60
					# Map x to location in control area
					xControl = self.screenWidth - int((x - 40) * (self.screenWidth / 20))
					cv2.circle(self.controlArea, (xControl, 50), 20, self.colour, -1)

				# Calculate average eye aspect ratio
				averageEyeRatio = (leftEyeRatio + rightEyeRatio) / 2.0

				leftCorner = leftEye[0]
				rightCorner = leftEye[3]

				angle = math.atan((leftCorner[1] - rightCorner[1]) / (leftCorner[0] - rightCorner[0]))

				# Calculate blinkScore based on eye aspect ratio
				if averageEyeRatio < self.blinkThreshold:
					self.blinkScore += (self.blinkThreshold - averageEyeRatio)
					# print("{}".format(averageEyeRatio))
					self.blinkFrameCounter += 1

				# Not blinking, check for winks
				else:
					# Check if previous blink score passes threshold
					if self.blinkScore > 0.05:
						self.blinks += 1

						# Check if blink was long blink
						if self.blinkFrameCounter >= self.blinkFrameThresh:
							self.blink()

					self.blinkFrameCounter = 0
					self.blinkScore = 0

				# Check for left eye wink
				# if leftEyeRatio < self.blinkThreshold and (rightEyeRatio - leftEyeRatio) > 0.025:
				if eyebrowToEyeDistanceRatio < 0.95:
					self.leftEyeWinkCounter += 1
					if self.leftEyeWinkCounter >= self.blinkFrameThresh:
						self.leftWink()
				else:
					self.leftEyeWinkCounter = 0

				# Check for right eye wink
				# if rightEyeRatio < self.blinkThreshold and (leftEyeRatio - rightEyeRatio) > 0.025:
				if eyebrowToEyeDistanceRatio > 1.05:
					self.rightEyeWinkCounter += 1
					if self.rightEyeWinkCounter >= self.blinkFrameThresh:
						self.rightWink()
				else:
					self.rightEyeWinkCounter = 0

				# Get bounding box coordinates
				(x, y, w, h) = face_utils.rect_to_bb(face)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

				cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (255, 0, 0), 1)
				cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (255, 0, 0), 1)


				# Display number of blinks
				cv2.putText(frame, "Blinks: {}".format(self.blinks), (10, 30),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				# Display average eye aspect ratio
				cv2.putText(frame, "Left Eye Ratio: {:.2}. Right Eye Ratio: {:.2}".format(leftEyeRatio, rightEyeRatio), (250, 30),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				# Draw facial landmarks
				for (x, y) in FaceLandmarks:
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

	# Function for when intential blink occurs
	def blink(self):
		# self.colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		self.colour = (0, 255, 0)
		print("BLINK")

	# Function for wink of left eye
	def leftWink(self):
		# Change colour to red
		self.colour = (255, 0, 0)
		print(self.colour)

	# Function for wink of right eye
	def rightWink(self):
		# Change colour
		self.colour = (0, 0, 255)
		print(self.colour)

if __name__ == '__main__':
	obj = EyeGaze()
	if len(sys.argv[:]) == 2:
		arg = sys.argv[1]
	else:
		arg = ""
	obj.start(arg)
