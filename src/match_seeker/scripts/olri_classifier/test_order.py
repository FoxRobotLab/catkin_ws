from paths import DATA
import numpy as np
import cv2

image = np.load(DATA + 'lstm_img_cell_Inpute.npy')
image = image[:,:,:,0]
label = np.load(DATA + 'lstm_heading_hotLabel.npy')

print("This is the length of image", len(image))

cnt = 0
while (cnt != 12001):
    img = image[cnt]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_text = cv2.putText(img, str(cnt), (10,10), font, 50, (0, 255, 0), 1)
    cv2.imshow("Window", img_text)
    cv2.waitKey(0)
    cnt +=1
cv2.destroyAllWindows()
