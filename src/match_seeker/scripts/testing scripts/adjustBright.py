

import os
import cv2

def setupMatcher():
    # FLANN_INDEX_KDTREE = 1
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,
    #                     key_size=12,
    #                     multi_probe_level=1)
    # search_params = dict(checks=50)
    # matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matcher = cv2.BFMatcher(cv2.HAMMING_NORM_TYPE, crossCheck=True)
    return matcher


def doMatch(matcher, des1, des2):
    matches = matcher.match(des1, des2)
    goodMatches = [m for m in matches if m.distance < 35]
    # print "  All matches: ", len(matches), "Good matches:", len(goodMatches)
    return goodMatches


def featuresAndMatching(im1, im2, imNum1, imNum2, windowName):
    global matcher

    textTemplate = "Img {0:0>4d} and Img {1:0>4d}: {2:d}"
    kp1, des1 = orb.detectAndCompute(im1, None)
    keypIm1 = cv2.drawKeypoints(im1, kp1, None)

    kp2, des2 = orb.detectAndCompute(im2, None)
    keypIm2 = cv2.drawKeypoints(im2, kp2, None)

    goodMatches = doMatch(matcher, des1, des2)
    # selectedGood.sort(key = lambda a: a.distance)
    numGood = len(goodMatches)
    # if numGood > 30:
    #     prString = textTemplate.format(imNum1, imNum2, numGood)
    #     matIm = cv2.drawMatches(keypIm1, kp1, keypIm2, kp2, goodMatches, None)
    #     cv2.putText(matIm, prString, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0))
    #     cv2.imshow(windowName, matIm)
    return numGood

def waitForUser():
    val = cv2.waitKey()
    ch = chr(val % -xFF)
    return ch



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create()
matcher = setupMatcher()
pathToImages = '/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/catkin_ws/src/match_seeker/res/create060717/'
# imNames = ['frame1171.jpg', 'frame1173.jpg', 'frame1199.jpg', 'frame1205.jpg']
# imNames = ['frame0953.jpg', 'frame0954.jpg', 'frame0955.jpg', 'frame0956.jpg']

print "--------------- Reading Images ------------------"
imNames = os.listdir(pathToImages)
images = []
eqImages = []
for name in imNames:
    nextIm = cv2.imread(pathToImages + name)
    if nextIm is not None:
        images.append(nextIm)
        alt = cv2.cvtColor(nextIm, cv2.COLOR_BGR2YCrCb)
        [br, cr, cb] = cv2.split(alt)
        # newBr = cv2.equalizeHist(br)
        newBr = clahe.apply(br)
        eqOdd = cv2.merge([newBr, cr, cb])
        eqIm1 = cv2.cvtColor(eqOdd, cv2.COLOR_YCrCb2BGR)
        eqIm2 = cv2.bilateralFilter(eqIm1, 5, 75, 75)
        eqImages.append(eqIm2)

        if len(images) % 100 == 0:
            print "."

ch = 'a'
outf = open("origEqMatchResults.txt", 'w')

for i in range(len(images) - 1):
    for j in range(i+1, len(images)):
        firstIm = images[i]
        firstEq = eqImages[i]
        secondIm = images[j]
        secondEq = eqImages[j]

        score1 = featuresAndMatching(firstIm, secondIm, i, j, "Orig Matches")
        score2 = featuresAndMatching(firstEq, secondEq, i, j, "Equalized Matches")
        if score1 > 30 or score2 > 30:
            outf.write(str((i, j)) + " BOTH GOOD,  Orig score = " + str(score1) + "  Eq score = " + str(score2) + "\n")
        if score1 > 30 and score2 <= 30:
            outf.write(str((i, j)) + " BAD EQUAL,  Orig score = " + str(score1) + "  Eq score = " + str(score2) + "\n")
        elif score1 <= 30 and score2 > 30:
            outf.write(str((i, j)) + " BAD ORIGI,  Orig score = " + str(score1) + "  Eq score = " + str(score2) + "\n")
        if ch == 'q':
            break
    if ch == 'q':
        break

outf.close()

cv2.destroyAllWindows()
