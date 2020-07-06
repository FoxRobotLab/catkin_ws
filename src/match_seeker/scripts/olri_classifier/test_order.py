from paths import DATA
import numpy as np
import cv2


cell_frame = np.load(DATA+ 'cell_origframes_500orL.npy',allow_pickle='TRUE').item()
print(cell_frame)
print(cell_frame.keys())
# image = np.load(DATA + 'lstm_img_cell_Inpute.npy')
# image = image[:,:,:,0]
# label = np.load(DATA + 'lstm_heading_hotLabel.npy')
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
