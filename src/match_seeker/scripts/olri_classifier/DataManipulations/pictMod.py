from src.match_seeker.scripts.olri_classifier.paths import DATA
import cv2

imagePath = DATA + 'frames/moreframes/frame'

for i in range(20603,28462, 1):
    img = cv2.imread(imagePath + str(i)+ '.jpg')
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

