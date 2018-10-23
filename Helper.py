from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math
import sys


class Helper:

	def eyeRatio(self, eye):
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

	def eyeRegion(self, eye, frame):
		xmax, ymax = eye[0]
		xmin, ymin = eye[0]
		for (x, y) in eye:
			if x > xmax:
				xmax = x
			if x < xmin:
				xmin = x
			if y > ymax:
				ymax = y
			if y < ymin:
				ymin = y

		return frame[ymin:ymax, xmin:xmax]

	def gammaCorrection(self, image, gamma):
		# Lookup table to
		lookupTable = np.array([((i / 255.0) ** gamma) * 255
		                        for i in np.arange(0, 256)]).astype("uint8")

		gammaCorrected = cv2.LUT(image, lookupTable)

		return gammaCorrected

	def get_iris_center(self, image, eye, gamma=1.0):
		# Increased size to width of 100 for better results
		image = imutils.resize(image, width=100)
		# Convert to grayscale
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Gamma correction
		img = self.gammaCorrection(img, gamma)

		# Return to caller to adjust gamma in future call
		meanAfterGamma = cv2.mean(img)[0]

		img = cv2.GaussianBlur(img, (5, 5), 0)

		# Normalise image after gamma correction
		cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

		cv2.imshow("{} normalised".format(eye), img)

		# Binarise image
		_, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

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
		# X and Y coordinates of centre of iris
		cX = cY = 0

		cv2.imshow("{} Eye threshold".format(eye), thresh)

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]

		# Check if contour is found
		if len(cnts) == 0:
			return meanAfterGamma, (0, 0)
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
			if circularity > 1.5 or circularity < 0.70:
				return meanAfterGamma, (0, 0)

			cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)

			cv2.drawContours(image, [c], 0, (0, 255, 0), 1)

		cv2.imshow("{} Eye".format(eye), image)

		cv2.waitKey(1)

		# Return mean intensity after gammma correction,
		# relative x, y coordinates of the centre of the iris
		return meanAfterGamma, (cX / 100, cY / len(image[:]))
