from paths import DATA
import numpy as np
import cv2
import collections

labels= np.load(DATA+ 'lstm_Heading_Output.npy')
print("This is the length", len(labels))

for i in range(500):
    print(i, labels[i])


#image = image[:,:,:,0]
#
# print("This is the length of image", len(image))
#
# cnt = 480
# while (cnt != 12001):
#     img = image[cnt]
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     img = cv2.putText(np.float32(img), str(cnt), (10,50), font, 1, (255, 255, 255), 5, cv2.LINE_AA)
#     cv2.imshow("Window", img)
#     cv2.waitKey(0)
#     cnt +=1
# cv2.destroyAllWindows()
