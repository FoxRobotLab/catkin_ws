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
    print "All matches size:", len(matches)
    goodMatches = []
    for mPair in matches:
        if len(mPair) > 1:
            [m, n] = mPair
            if m.distance < 0.7 * n.distance:
                goodMatches.append(m)
        else:
            goodMatches.append(mPair[0])
    print "Good matches size:", len(goodMatches)
    return goodMatches


def featuresAndMatching(im1, im2, windowName):
    kp1, des1 = orb.detectAndCompute(im1, None)
    keypIm1 = cv2.drawKeypoints(im1, kp1, None)

    kp2, des2 = orb.detectAndCompute(im2, None)
    keypIm2 = cv2.drawKeypoints(im2, kp2, None)

    goodMatches = doFLANN(flann, des1, des2)

    matIm1 = cv2.drawMatches(keypIm1, kp1, keypIm2, kp2, goodMatches, None)
    cv2.imshow(windowName, matIm1)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create()
flann = setupFLANN()

# imNames = ['frame1171.jpg', 'frame1173.jpg', 'frame1199.jpg', 'frame1205.jpg']
imNames = ['frame0953.jpg', 'frame0954.jpg', 'frame0955.jpg', 'frame0956.jpg']
images = []
for name in imNames:
    nextIm = cv2.imread(name)
    images.append(nextIm)


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

ch = 'a'
for i in range(len(images) - 1):
    for j in range(i+1, len(images)):
        print ("============", (i, j), "==========")
        firstIm = images[i]
        firstEq = eqImages[i]
        secondIm = images[j]
        secondEq = eqImages[j]

        print "Orig images matched:"
        featuresAndMatching(firstIm, secondIm, "Orig Matches")
        print "Equalized images matched"
        featuresAndMatching(firstEq, secondEq, "Equalized Matches")
        print "Original 1 matched against equalized"
        featuresAndMatching(firstIm, firstEq, "Orig and its Eq 1")
        print "Original 2 matched against equalized"
        featuresAndMatching(secondIm, secondEq, "Orig and its Eq 2")
        val = cv2.waitKey()
        ch = chr(val % 0xFF)
        if ch == 'q':
            break
    if ch == 'q':
        break


cv2.destroyAllWindows()
