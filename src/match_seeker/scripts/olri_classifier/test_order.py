from paths import DATA
import numpy as np
from collections import OrderedDict





master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'

def getFrameHeadingDict():
    fhd = {}
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            fhd['%04d'%int(split[0])] = split[-1]

    return fhd

if __name__ == '__main__':
    labels= np.load(DATA+ 'lstm_Heading_Output.npy')
    cell_frame_dict = np.load(DATA+ 'cell_origFrames.npy',allow_pickle='TRUE').item()
    frame_label = getFrameHeadingDict()

    ######################The actual hot labels!!!!!!!!!
    cel = '22'
    start = (int(cel)-18)*500
    hotLabel = []
    for i in range(start, start+500, 1):
        hotLabel.append(labels[i])
    ##########################

    #########################The frames
    wantedCells = ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                   '35', '36', '37', '38', '39', '40', '41', '42']
    frame_dict = OrderedDict()
    for cell in wantedCells:
        frame_dict[cell] = cell_frame_dict[cell]
    #####################


    ############checking that they are the same
    which = 0
    for frame in frame_dict[cel]:
        head = frame_label[frame]
        onehot = [0] * 8
        onehot[int(head)//45] = 1
        onehot = np.asarray(onehot)
        print(onehot)
        print(hotLabel[which])
        if onehot != hotLabel[which]:
            print(frame)
        which +=1










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
