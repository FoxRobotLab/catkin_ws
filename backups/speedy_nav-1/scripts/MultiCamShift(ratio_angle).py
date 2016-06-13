import numpy as np
import cv2
import threading
import math
import itertools

from TargetScanner import *

class MultiCamShift(threading.Thread):

	def __init__(self, exampleImage):
		"""Creates the cam shift thread and sets up scanners for all objects listed in 'self.toScanFor'. Needs an example image to get the dimensions of the frame."""

		threading.Thread.__init__(self)
		self.toScanFor = ["purple","green","red","blue"]
		self.scanners = {}
		self.lock = threading.Lock()
		self.locationAndArea = {}

		if exampleImage == None:
			cap = cv2.VideoCapture(0)
			ret, exampleImage = cap.read()
			cap.release()

		self.fHeight, self.fWidth, self.fDepth = exampleImage.shape

		for object_name in self.toScanFor:
			self.scanners[object_name] = TargetScanner(object_name, (self.fWidth, self.fHeight))

		# determines how close together in the y direction the colors of a horizontal pattern must be.
		self.horzPatternYSpacing = self.fHeight / 8.0
		# determines how uniform the distances between the colors in a pattern must be
		self.horzPatternXRatio = 2


	def run(self, vid_src):
		"""Will run the tracking program on the video from vid_src."""

		self.vid = cv2.VideoCapture(vid_src)
		ret, frame = self.vid.read()

		cv2.namedWindow("MultiTrack")

		while(ret):
			self.update(frame)

			cv2.imshow("MultiTrack", frame)

			char = chr(cv2.waitKey(50) & 255)
			if char == "0":
				break

			ret,frame = self.vid.read()

		self.close()


	def update(self, image):
		"""Updates the trackers with the given image."""

		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_image, np.array((0., 60., 32.)), np.array((255., 255., 255.)))

		objects = {}

		for object_name in self.scanners:
			scanner = self.scanners[object_name]
			image = scanner.scan(image, hsv_image, mask)
			objects[object_name] = scanner.getTrackingInfo()

		with self.lock:
			self.locationAndArea = objects

		return image


	def getObjectsOfColor(self, color_name):
		"""Returns a list of objects locations and area of all identified objects of type 'color_name'."""

		with self.lock:
			locationAndArea = self.locationAndArea[color_name]
		return locationAndArea


	def getAverageOfColor(self, color_name):
		"""Returns the average location and sum of area of all identified objects of type 'color_name'."""
		with self.lock:
			dataList = self.locationAndArea[color_name].copy()
		xTotal = 0
		yTotal = 0
		aTotal = 0.0
		for data in dataList:
			((x,y), a) = data
			xTotal += x
			yTotal += y
			aTotal += a
		size = len(dataList)
		return [((xTotal / size, yTotal / size), aTotal)] if size != 0 else []


	def getHorzPatterns(self):
		"""Searches for patterns of three colors placed horizontally. It returns the most likely match with information about the pattern, location in frame, porportion of the screen taken up by the pattern, and the predicted angle of the robot to the pattern"""
		with self.lock:
			tracked = self.locationAndArea.copy()
		colors = tracked.keys()
		for color in colors[:]:
			if tracked[color] == []:
				colors.remove(color)
		if len(colors) < 3:
			 return None
		# gets all possible combinations of three colors of length 3
		colorCombos = list(itertools.combinations(colors, 3))
		retVal = None
		for combo in colorCombos:
			c1,c2,c3 = combo
			c1Tracked = tracked[c1] # (x,y) coords of tracked object
			c2Tracked = tracked[c2]
			c3Tracked = tracked[c3]
			for c1_obj in c1Tracked:
				c1_obj_tup = c1_obj, c1 # creates a tuple of the tracked object and its color
				for c2_obj in c2Tracked:
					c2_obj_tup = c2_obj, c2
					for c3_obj in c3Tracked:
						c3_obj_tup = c3_obj, c3
						xCoords = (c1_obj[0], c2_obj[0], c3_obj[0])
						yCoords = (c1_obj[1], c2_obj[1], c3_obj[1])
						if self.checkXCoords(xCoords) and self.checkYCoords(yCoords):
							sort = sorted((c1_obj_tup, c2_obj_tup, c3_obj_tup))
							sortedCombo = tuple(x[1] for x in sort)
							leftColor, centerColor, rightColor = sort
							lx,ly,lw,lh = leftColor[0]
							cx,cy,cw,ch = centerColor[0]
							rx,ry,rw,rh = rightColor[0]
							lArea = lw * lh
							cArea = cw * ch
							rArea = rw * rh
							dxList = [cx - lx, rx - cx]
							dxRatio = max(dxList) / float(min(dxList))
							averageRatio = (lw / float(lh) + cw / float(ch) + rw / float(rh)) / 3.0
							centerRatio = (cw / float(ch))
							centerRatio = max(0, min(centerRatio, 2)) # ratio between 0 and -2
#							angle = 90 - math.degrees(math.asin(averageRatio / 2))
#							angle = 90 - math.degrees(math.asin((cw / float(ch)) / 2))
#							angle = 90 - averageRatio * 45
#							angle = 90 - (cw / float(ch)) * 45

							#eliptical graph of botHeight 90 and botWidth 2
							angle = 45 * math.sqrt(4 - x^2)

							if lArea > rArea:
								angle *= -1
							angle = min(90, max(-90, angle)) # constrains angle to the range -90 to 90
							totalArea = lArea + cArea + rArea
							relativeArea = totalArea / float(self.fWidth * self.fHeight)
							if retVal == None or dxRatio < retVal[1]:
 								retVal = ((sortedCombo, (cx, cy), relativeArea, angle),
										dxRatio)
		return None if not retVal else retVal[0]


	def checkXCoords(self, xCoords):
		"""Checks to see if the x distances are somewhat consitant"""
		xCoords = sorted(xCoords)
		dXs = xCoords[1] - xCoords[0], xCoords[2] - xCoords[1]
		if max(dXs) / (min(dXs) + 1) < self.horzPatternXRatio:
			return True
		return False


	def checkYCoords(self, yCoords):
		"""Checks to see if the y coords are close together"""
		yCoords = sorted(yCoords)
		if yCoords[2] - yCoords[0] <= self.horzPatternYSpacing:
			return True
		return False


	def close(self):
		"""Closes the program."""

		cv2.destroyAllWindows()
		self.vid.release()


	def getPotentialMatches(self):
		"""Finds groups of two different colors that are close by and returns the average location of all potential patterns."""
		with self.lock:
			data = self.locationAndArea.copy()
		colors = data.keys()
		for color in colors[:]:
			if data[color] == []:
				colors.remove(color)
		if len(colors) < 2:
			return None
		potMatches = []
		combos = itertools.combinations(colors)
		for combo in combos:
			c1, c2 = combo
			c1Tracked, c2Tracked = data[c1], data[c2]
			for c1Location in c1Tracked:
				c1x, c1y, c1w, c1h = c1Location
				for c2Location in c2Tracked:
					c2x, c2y, c2w, c2h = c2Location
					if abs(c1x-c2x) < self.fWidth / 3.0 and abs(c1y - c2y) < self.fHeight / 6.0:
						x = (c1x + c2x) / 2
						y = (c1y + c2y) / 2
						potMatches.append((x,y))
		xSum, ySum = 0, 0
		for potMatch in potMatches:
			x, y = potMatch
			xSum += x
			ySum += y
		numElems = len(potMatches)
		return None if numElems == 0 else (xSum / numElems, ySum / numElems)


	def getFrameShape(self):
			"""Returns the the dimmensions and depth of the camera frame"""
			return self.fWidth, self.fHeight, self.fDepth

	def getFrameCenter(self):
			"""Returns the center coordinates of the camera frame"""
			return self.fWidth/2, self.fHeight/2


























