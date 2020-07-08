from paths import DATA
import numpy as np
from collections import OrderedDict






if __name__ == '__main__':
    image_cell = np.load(DATA+ 'lstm_Img_Cell_Input13k.npy')
    print(image_cell.shape)
    # cells = []
    # for cell in image_cell:
    #     print(cell[1])
    #     cells.append(cell[1])
    # print(cells)
    # print("this is how many cells there are", len(cells))







    # labels= np.load(DATA+ 'lstm_Heading_Output.npy')
    # cell_frame_dict = np.load(DATA+ 'cell_origFrames.npy',allow_pickle='TRUE').item()
    # frame_label = getFrameHeadingDict()
    #
    # ######################The actual hot labels!!!!!!!!!
    # cel = '29'
    # start = (int(cel)-18)*500
    # hotLabel = []
    # for i in range(start, start+500, 1):
    #     hotLabel.append(labels[i])
    # ##########################
    #
    # #########################The frames
    # wantedCells = ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
    #                '35', '36', '37', '38', '39', '40', '41', '42']
    # frame_dict = OrderedDict()
    # for cell in wantedCells:
    #     frame_dict[cell] = cell_frame_dict[cell]
    # #####################
    #
    #
    # ############checking that they are the same
    # which = 0
    # for frame in frame_dict[cel]:
    #     head = frame_label[frame]
    #     onehot = [0] * 8
    #     onehot[int(head)//45] = 1
    #     onehot = np.asarray(onehot)
    #     if (onehot != hotLabel[which]).all():
    #         print(frame)
    #     which +=1
    #
    #
    #
    #
    #
    #
    #



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
