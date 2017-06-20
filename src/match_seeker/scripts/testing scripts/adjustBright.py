

import os
import cv2

def setupFLANN():
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flanner = cv2.FlannBasedMatcher(index_params, search_params)
    return flanner


def doFLANN(flanner, des1, des2):
    matches = flann.knnMatch(des1, des2, k=2)
    goodMatches = []
    for mPair in matches:
        if len(mPair) > 1:
            [m, n] = mPair
            if m.distance < 0.7 * n.distance:
                goodMatches.append(m)
        elif len(mPair) == 1:
            goodMatches.append(mPair[0])
    print "  All matches: ", len(matches), "Good matches:", len(goodMatches)
    return goodMatches


def featuresAndMatching(im1, im2, imNum1, imNum2, windowName):
    textTemplate = "Img {0:0>4d} and Img {1:0>4d}: {2:d}"
    kp1, des1 = orb.detectAndCompute(im1, None)
    keypIm1 = cv2.drawKeypoints(im1, kp1, None)

    kp2, des2 = orb.detectAndCompute(im2, None)
    keypIm2 = cv2.drawKeypoints(im2, kp2, None)

    goodMatches = doFLANN(flann, des1, des2)
    selectedGood = [m for m in goodMatches if m.distance < 35]
    # selectedGood.sort(key = lambda a: a.distance)
    numGood = len(selectedGood)
    print "Number of good matches =", numGood
    if numGood > 30:
        prString = textTemplate.format(imNum1, imNum2, numGood)
        matIm = cv2.drawMatches(keypIm1, kp1, keypIm2, kp2, goodMatches, None)
        cv2.putText(matIm, prString, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0))
        cv2.imshow(windowName, matIm)
    return numGood

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create()
flann = setupFLANN()
pathToImages = '/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/catkin_ws/src/match_seeker/res/create060717/'
# imNames = ['frame1171.jpg', 'frame1173.jpg', 'frame1199.jpg', 'frame1205.jpg']
# imNames = ['frame0953.jpg', 'frame0954.jpg', 'frame0955.jpg', 'frame0956.jpg']

print "--------------- Reading Images ------------------"
imNames = os.listdir(pathToImages)
images = []
for name in imNames:
    nextIm = cv2.imread(pathToImages + name)
    if nextIm is not None:
        images.append(nextIm)
        if len(images) % 100 == 0:
            print "."

print "--------------- Equalizing ------------------"
eqImages = []
for i in range(len(images)):
    alt = cv2.cvtColor(images[i], cv2.COLOR_BGR2YCrCb)
    [br, cr, cb] = cv2.split(alt)
    # newBr = cv2.equalizeHist(br)
    newBr = clahe.apply(br)
    eqOdd = cv2.merge([newBr, cr, cb])
    eqIm1 = cv2.cvtColor(eqOdd, cv2.COLOR_YCrCb2BGR)
    eqIm2 = cv2.bilateralFilter(eqIm1, 5, 75, 75  )
    eqImages.append(eqIm2)
    if i % 100 == 0:
        print "."

ch = 'a'
for i in range(len(images) - 1):
    for j in range(i+1, len(images)):
        print "============", (i, j), "=========="
        firstIm = images[i]
        firstEq = eqImages[i]
        secondIm = images[j]
        secondEq = eqImages[j]

        score1 = featuresAndMatching(firstIm, secondIm, i, j, "Orig Matches")
        score2 = featuresAndMatching(firstEq, secondEq, i, j, "Equalized Matches")
        if score1 > 30 or score2 > 30:
            print "Orig images matched:", score1
            print "Equalized images matched", score2
            val = cv2.waitKey()
            ch = chr(val % 0xFF)
            if ch == 'q':
                break
    if ch == 'q':
        break


cv2.destroyAllWindows()
