# Facial landmarks

from Helper import *
import sys
import cv2
import numpy as np
import imutils
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import math
import random
from collections import deque
import time


class EyeGaze:
	"""
	Initialize the class with all the required Variables.
	"""
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

		# fatigue Detection
		self.fatigueThresh = 0.24
		self.fatigueCounter = 0
		self.fatigueDetection = 0
		self.AvgEAROvrTime = 0

		# Position of centre of eye corresponding to looking at left and right sides of screen
		self.leftValue = 60
		self.rightValue = 40

		self.helper = Helper()

		# Face detector
		self.detector = dlib.get_frontal_face_detector()

		# Facial landmark predictor is used to find facial features such as eyes and nose
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		# Initial gamma
		self.gammaLeft = 1.0
		self.gammaRight = 1.0

		# Initial mean intensity of eye region
		self.meanLeft = 0
		self.meanRight = 0

		# Upper and lower bounds of mean
		self.meanUpper = 80
		self.meanLower = 75

		self.screenWidth = 1400
		self.controlArea = np.ones((100, self.screenWidth, 3), np.uint8) * 255

		# Stores the latest 5 gaze locations
		self.gazeLocation = deque(maxlen=5)

		# Circle colour
		self.colour = (0,255,0)

		# Log ear data for fatigue monitoring
		self.earDataFile = open("earData.txt", "w+")

	"""
	Driver function for eye gaze detection.
	"""
	def start(self, arg=""):
		# Use sample video
		if len(arg) != 0:
			camera = cv2.VideoCapture(arg)
		else:
			camera = cv2.VideoCapture(0)

		accuracy = 0

		while camera.isOpened():
			# Get frame from webcam
			ret, frame = camera.read()

			if ret == False:
				break

			# Draw area numbers
			cv2.putText(self.controlArea, "1", (140, 50),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
			cv2.putText(self.controlArea, "2", (420, 50),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
			cv2.putText(self.controlArea, "3", (700, 50),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
			cv2.putText(self.controlArea, "4", (980, 50),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
			cv2.putText(self.controlArea, "5", (1260, 50),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
			# Draw lines
			cv2.line(self.controlArea, (280, 0), (280, 100), (0,0,0), 2)
			cv2.line(self.controlArea, (560, 0), (560, 100), (0,0,0), 2)
			cv2.line(self.controlArea, (840, 0), (840, 100), (0,0,0), 2)
			cv2.line(self.controlArea, (1120, 0), (1120, 100), (0,0,0), 2)
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
				# print("xL = {}, xR = {}".format(int(xL * 100), int(xR * 100)))

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

					if x < self.rightValue:
						x = self.rightValue
					if x > self.leftValue:
						x = self.leftValue

					# Map x to location in control area
					xControl = self.screenWidth - int((x - self.rightValue) * (self.screenWidth / (self.leftValue - self.rightValue)))
					# Add gaze location to deque
					self.gazeLocation.append(xControl)
					# Find mean of 5 most recent gaze locations
					meanGaze = int(np.mean(self.gazeLocation))

					diff = abs(420 - meanGaze)

					accuracy += diff
					accuracy = accuracy / 2

					cv2.circle(self.controlArea, (meanGaze, 50), 20, self.colour, -1)

				# Calculate average eye aspect ratio
				averageEyeRatio = (leftEyeRatio + rightEyeRatio) / 2.0

				# update average eye aspect ratio over time
				self.updateAvgEAROvrTime(averageEyeRatio)

				# check if fatigue is detected
				if self.AvgEAROvrTime < self.fatigueThresh:
					if self.fatigueDetection > 10:
						self.raiseAlarm()
						self.fatigueDetection = 0
					self.fatigueDetection += 1
					self.fatigueCounter = 0
				else:
					self.fatigueDetection = 0

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

				# print(eyebrowToEyeDistanceRatio)
				# Check for left eye wink
				# if leftEyeRatio < self.blinkThreshold and (rightEyeRatio - leftEyeRatio) > 0.025:
				if eyebrowToEyeDistanceRatio < 0.967:
					self.leftEyeWinkCounter += 1
					if self.leftEyeWinkCounter >= self.blinkFrameThresh:
						self.leftWink()
						self.leftEyeWinkCounter = 0
				else:
					self.leftEyeWinkCounter = 0

				# Check for right eye wink
				# if rightEyeRatio < self.blinkThreshold and (leftEyeRatio - rightEyeRatio) > 0.025:
				if eyebrowToEyeDistanceRatio > 1.07:
					self.rightEyeWinkCounter += 1
					if self.rightEyeWinkCounter >= self.blinkFrameThresh:
						self.rightWink()
						self.rightEyeWinkCounter = 0
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

				cv2.putText(frame, "Leye Winks: {}".format(self.leftEyeWinkCounter), (10, 60),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				cv2.putText(frame, "Reye winks: {}".format(self.rightEyeWinkCounter), (10, 90),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				# Display average eye aspect ratio
				cv2.putText(frame, "Left Eye Ratio: {:.2}.".format(leftEyeRatio), (400, 30),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				cv2.putText(frame, "Left Eye Ratio: {:.2}.".format(rightEyeRatio), (400, 60),
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

		print("acc = {}".format(accuracy))

		camera.release()
		self.earDataFile.close()
		cv2.destroyAllWindows()
		# webcam is still in use if python doesn't quit
		quit()

	def calibrate(self, arg=""):
		# Use sample video
		if len(arg) != 0:
			camera = cv2.VideoCapture(arg)
		else:
			camera = cv2.VideoCapture(0)

		leftLook = False
		leftAvg = 0
		rightLook = False
		rightAvg = 0
		rightCount = 0
		leftCount = 0

			
		cv2.imshow("Control Window", self.controlArea)

		print("CALIBRATION")
		print("Centre the control window at the top of your screen")
		print("Select the control window before entering commands")
		print("When looking at the left edge of the control window,\npress 'l' to begin calibration of left eye")

		while camera.isOpened():
			# Get frame from webcam
			ret, frame = camera.read()

			if ret == False:
				break

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

				# Get coordinates of 6 eye features
				leftEye = FaceLandmarks[lStart:lEnd]
				rightEye = FaceLandmarks[rStart:rEnd]

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


				# Pupil located for both eyes
				if xL > 0 and xR > 0:
					x = int((xL + xR) * 50)

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
				# Record pupil position for mean calculation
				if x != 0:
					if cv2.waitKey(10) & 0xFF == ord('l'):
						print("Record Left")
						if leftLook == False:
							
							leftLook = True
							rightLook = False

					if cv2.waitKey(10) & 0xFF == ord('r'):
						if rightLook == False:
							print("Record Right")
							rightLook = True


					if leftLook:
						leftCount += 1
						leftAvg += x
						leftAvg = leftAvg / 2
						# Find average pupil position for 60 frames
						if leftCount > 60:
							print("LEFT CALIBRATED")
							print("When looking at the right edge of the control window,\npress 'r' to begin calibration of right eye")
							self.leftValue = int(leftAvg)
							leftLook = False

					if rightLook:
						rightCount += 1
						rightAvg += x
						rightAvg = rightAvg / 2
						# Find average pupil position for 60 frames
						if rightCount > 60:
							print("RIGHT CALIBRATED")
							print("Press 'q' to start control")
							rightLook = False
							self.rightValue = int(rightAvg)

			cv2.imshow("WebCam", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		print("L = {} R = {}".format(self.leftValue, self.rightValue))

		camera.release()
		cv2.destroyAllWindows()
		# webcam is still in use if python doesn't quit
		# quit()

	"""
	Function to achieve the functionality of blinks
	"""
	def blink(self):
		# self.colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		# self.colour = (0, 255, 0)
		# print("BLINK")
		pass

	"""
	Function to achieve the functionality of blinks
	"""
	def leftWink(self):
		# Change colour to red
		self.colour = (255, 0, 0)

	"""
	Function to achieve the functionality of blinks
	"""
	def rightWink(self):
		# Change colour
		self.colour = (0, 0, 255)

	"""
	Function to update the Average EAR over time
	"""
	def updateAvgEAROvrTime(self, EAR):
		sum = self.AvgEAROvrTime * self.fatigueCounter + EAR
		# print(self.fatigueCounter)
		self.fatigueCounter += 1
		self.AvgEAROvrTime = sum / self.fatigueCounter
		if self.fatigueCounter > 100:
			print(self.AvgEAROvrTime)
			self.earDataFile.write("{} ".format(self.AvgEAROvrTime))
			self.fatigueCounter = 0

	"""
	Function to raise an alert/alarm 
	that the user is experiencing fatigue.
	"""
	def raiseAlarm(self):
		print("Fatigue Detected")

if __name__ == '__main__':
	obj = EyeGaze()
	if len(sys.argv[:]) == 2:
		arg = sys.argv[1]
	else:
		arg = ""
	obj.calibrate(arg)
	time.sleep(1.0)
	obj.start(arg)