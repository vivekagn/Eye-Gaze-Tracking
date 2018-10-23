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
		self.blinkThreshold = 0.3
		self.blinkFrameThresh = 3
		self.blinkFrameCounter = 0
		self.leftEyeWinkCounter = 0
		self.rightEyeWinkCounter = 0
		self.blinks = 0
		self.blinkScore = 0

		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		self.helper = Helper()
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		self.color = (0,225,0)

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
		if len(arg)!=0:
			camera = cv2.VideoCapture(arg)

		else:
			camera = cv2.VideoCapture(0)

		while camera.isOpened():
			ret, frame = camera.read()

			cv2.imshow("Control Area", self.controlArea)

			if ret == False:
				break

			xL = yL = xR = yR = 0

			# Used for FPS calculation
			startTime = cv2.getTickCount()

			# Load frame from webcam, resize and convert to grayscale
			frame = imutils.resize(frame, width=600)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Detect faces in frame
			faces = self.detector(gray, 1)

			# loop over the face detections
			for (i, face) in enumerate(faces):
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = self.predictor(gray, face)
				shape = face_utils.shape_to_np(shape)

				# grab the indexes of the facial landmarks for the left and
				# right eye, respectively
				(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
				(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

				# Get coordinates of 6 eye features
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				# Calulate ratio of eyes
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
						self.gammaLeft -= 0.02
					elif self.meanLeft < self.meanLower:
						self.gammaLeft += 0.02
					meanL, (xL, yL) = self.helper.get_iris_center(lEye, "Left", gamma=self.gammaLeft)
					self.meanLeft += meanL
					self.meanLeft = self.meanLeft / 2

				if rightEyeRatio > 0.25:
					# Adjust gamma to suit brightness
					if self.meanRight > self.meanUpper:
						self.gammaRight -= 0.02
					elif self.meanRight < self.meanLower:
						self.gammaRight += 0.02
					meanR, (xR, yR) = self.helper.get_iris_center(rEye, "Right", gamma=self.gammaRight)
					self.meanRight += meanR
					self.meanRight = self.meanRight / 2

				# print("{} {}".format(mean,gamma))
				print("xL = {}, xR = {}".format(int(xL), int(xR)))

				if xL > 0 and xR > 0:
					x = int((xL + xR) * 50)
					print(
						"Left x: {}, y: {} Right x: {}, y: {}".format(int(xL * 100), int(yL * 100), int(xR * 100),
						                                              int(yR * 100)))
				elif xL > 0:
					x = int(xL * 100)
				elif xR > 0:
					x = int(xR * 100)
				else:
					x = 0

				if x != 0:
					# Clear control area
					self.controlArea[:, :, :] = 255
					# 35, 60 are locations in eye region corresponding to looking left and right on screen
					# Will need to add in a calibration so these aren't hard coded in
					if x < 40: x = 40
					if x > 60: x = 60
					# Map x to location in control area
					xControl = self.screenWidth - int((x - 40) * (self.screenWidth / 20))
					cv2.circle(self.controlArea, (xControl, 50), 20, self.color, -1)

				# Calculate average eye aspect ratio
				averageEyeRatio = (leftEyeRatio + rightEyeRatio) / 2.0

				leftCorner = leftEye[0]
				rightCorner = leftEye[3]

				angle = math.atan((leftCorner[1] - rightCorner[1]) / (leftCorner[0] - rightCorner[0]))

				# Calculate blinkScore based on eye aspect ratio
				if averageEyeRatio < self.blinkThreshold:
					self.blinkScore += (self.blinkThreshold - averageEyeRatio)
					self.blinkFrameCounter += 1
				
				# Not blinking, check for winks
				else:
					# Check for left eye wink
					if leftEyeRatio < self.blinkThreshold:
						self.leftEyeWinkCounter += 1
						if self.leftEyeWinkCounter >= self.blinkFrameThresh:
							self.leftWink()
					else:
						self.leftEyeWinkCounter = 0

					# Check for left eye wink
					if rightEyeRatio < self.blinkThreshold:
						self.rightEyeWinkCounter += 1
						if self.rightEyeWinkCounter >= self.blinkFrameThresh:
							self.rightWink()
					else:
						self.rightEyeWinkCounter = 0

					# Check if previous blink score passes threshold
					if self.blinkScore > 0.05:
						self.blinks += 1

						# Check if blink was long blink
						if self.blinkFrameCounter >= self.blinkFrameThresh:
							self.blink()

					self.blinkFrameCounter = 0
					self.blinkScore = 0

				# Get bounding box coordinates
				(x, y, w, h) = face_utils.rect_to_bb(face)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

				cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (255, 0, 0), 1)
				cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (255, 0, 0), 1)

				# show the face number
				cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				cv2.putText(frame, "Blinks: {}".format(self.blinks), (10, 30),
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

	def blink(self):
		# self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		# print(self.color)
		self.color = (0, 255, 0)
		print("BLINK")

	def leftWink(self):
		self.color = (255, 0, 0)
		print(self.color)

	def rightWink(self):
		self.color = (0, 0, 255)
		print(self.color)

if __name__ == '__main__':
	obj = EyeGaze()
	obj.start()