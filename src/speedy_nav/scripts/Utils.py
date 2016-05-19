import math



def getMedianAngle(leftDist, rightDist, angleBetween):
	"""Given two distances and the angle between them, finds the angle of the observer to the surface with flat against the wall facing left being zero."""
	if leftDist <= rightDist:
		shorter, longer = leftDist, rightDist
	else:
		shorter, longer = rightDist, leftDist
	
	shorter = float(shorter)
	longer = float(longer)
	angleBetween = float(math.radians(angleBetween))

	#Uses law of cosines to determine length of wall spanned by sensor
	wallDist = math.sqrt(math.pow(shorter,2) + math.pow(longer,2)
		- 2 * shorter * longer * math.cos(angleBetween))

	if wallDist <= 0:
		return 90
	
	#Uses law of sines to determine angle between wall and the shorter distance
	angleShort = math.asin(math.sin(angleBetween) * longer / wallDist)

	#Uses law of cosines to determine the length of median
	halfWall = wallDist / 2.0
	medianDist = math.sqrt(math.pow(shorter, 2) + math.pow(halfWall, 2)
		- 2 * shorter * halfWall * math.cos(angleShort))

	#Uses law of sines to determine the angle of the median to the wall
	medianAngle = math.asin(math.sin(angleShort) * shorter / medianDist)

	medianAngle = math.degrees(medianAngle)
	if leftDist < rightDist:
		medianAngle = 180 - medianAngle

	return medianAngle
