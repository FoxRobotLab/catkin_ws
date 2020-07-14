import numpy as np
import random
import cv2
from paths import DATA
from collections import OrderedDict

master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'

numCells = 271
image_size = 100
images_per_cell = 500

def getCellCounts():
    # Returns two dictionaries: cell_counts and cell_frame_dict. cell_counts has cells as keys and the number of total
    #frames as value. cell_frame_dict is an ordered dictionary with cells as keys and an array of associated frames as values
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        cell_counts = dict()
        cell_frame_dict = OrderedDict()
        for line in lines:
            splitline = line.split()
            if splitline[1] not in cell_counts.keys():
                if (len(cell_counts) >= numCells):
                    continue #DT
                cell_frame_dict[splitline[1]] = []
                cell_counts[splitline[1]] = 1
            else:
                cell_counts[splitline[1]] += 1
            cell_frame_dict[splitline[1]].append('%04d'%int(splitline[0]))
    return cell_counts, cell_frame_dict

def getUnderOverRep(cell_counts):
    #Returns two arrays, one with cells that have or are below 'images_per_cell' and the other with cells that have more
    #labels than 'images_per_cell'
    underRep = []
    overRep = []
    for key in cell_counts.keys():
        if int(cell_counts[key]) <= images_per_cell:
            underRep.append(key)
        else:
            overRep.append(key)
    return underRep, overRep

def getFrameCellDict():
    # Returns dict with cell corresponding to each frame
    frame_cell_dict = {}
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            frame_cell_dict['%04d'%int(split[0])] = split[1]
    return frame_cell_dict

def getHeadingRep(cell_counts):
    # Returns dict with cell as key and a matrix as value. There are 8 rows and 2 columns. The first column gives the heading
    #and the second column gives the total frames (within the corresponding cell/key) to have that heading
    cellHeadingCounts = dict()
    for key in cell_counts.keys():
        cellHeadingCounts[key] = [['0',0],['45',0],['90',0],['135',0],['180',0],['225',0],['270',0],['315',0]]
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            cell = str(split[1])
            if cell in cell_counts.keys():
                for head in cellHeadingCounts[cell]:
                    if head[0] == split[-1]:
                        head[1] += 1
                        break

    return cellHeadingCounts

def getHeadingFrameDict():
    #Creating a dictionary with heading as key and an array of all frames with that heading as value
    heading_frame_dict = {'0':[],'45':[],'90':[],'135':[],'180':[],'225':[],'270':[],'315':[]}

    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            heading_frame_dict[split[-1]].append('%04d'%int(split[0]))

    return heading_frame_dict

def cullOverRepped(cell_counts, cell_frame_dict, cell_heading_counts):
    #Takes all cells that have more than images_per_cell and randomly erases labels until the cell has exactly the same
    #number as images_per_cell
    under, overRepList = getUnderOverRep(cell_counts)
    heading_frame_dict = getHeadingFrameDict()
    i = 1
    for cell in overRepList:
        print('Cell '+ str(i) + " of " + str(len(overRepList)))
        i+=1
        print("This is the cell being modified", cell)
        while cell_counts[cell] > images_per_cell:
            headingList = sorted(cell_heading_counts[cell],key= lambda x: x[1])
            largestHeading = headingList[-1][0]
            headingList[-1][1] = headingList[-1][1] -1 #making sure that the biggest head count goes down by one
            potentialCulls = []
            for frame in heading_frame_dict[largestHeading]:
                if frame in cell_frame_dict[cell]:
                    potentialCulls.append(frame)
            toBeRemoved = potentialCulls[random.randint(0,len(potentialCulls)-1)]
            cell_frame_dict[cell].remove(toBeRemoved)
            cell_counts[cell] -= 1


def addUnderRepped(cell_counts, cell_frame_dict, cell_heading_counts):
    #Takes all cells that have below or the same amount of images_per_cell and keeps adding labels until it has the same
    #number of labels as images_per_cell
    underRepList, over = getUnderOverRep(cell_counts)
    heading_frame_dict = getHeadingFrameDict()
    rndUnderRepSubset = OrderedDict()
    i = 1
    for cell in underRepList:
        print('Cell '+ str(i) + " of " + str(len(underRepList)),cell)
        i+=1
        rndUnderRepSubset[cell] = []
        while cell_counts[cell] < images_per_cell:
            headingList = sorted(cell_heading_counts[cell],key= lambda x: x[1])
            h = 0
            while(headingList[h][1] == 0):
                h+=1
            smallestHeading = headingList[h][0]
            headingList[h][1] = headingList[h][1] + 1
            potentialAdditions = []
            for frame in heading_frame_dict[smallestHeading]:
                if frame in cell_frame_dict[cell]:
                    potentialAdditions.append(frame)
            if len(potentialAdditions) == 0:
                print(cell, 'has very little data')
                continue
            toBeAdded = random.choice(potentialAdditions)
            rndUnderRepSubset[cell].append(toBeAdded)

            cell_counts[cell] += 1
    #cullOverRepped must be run first. cell_frame_dict is updated in cullOverRepped and here as well. cell_frame_dict
    #has all cells, but not all cells have 500 frames. The ones that do not are also placed in rndUnderRepSubset where
    #random frames are chosen to complete the 500 frames.
    np.save(DATA+ 'cell_origFrames', cell_frame_dict)
    np.save(DATA + 'cell_newFrames', rndUnderRepSubset)
    return cell_frame_dict, rndUnderRepSubset

def resizeAndCrop(image):
    #Processing the frame into an image that is of 'image_size' and
    if image is None:
        print("No Image")
    else:
        cropped_image = cv2.resize(image, (image_size,image_size))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        return cropped_image

def getOneHotLabel(number,size):
    onehot = [0] * size
    onehot[number] = 1
    return onehot

def getFrameHeadingDict():
    #Dictionary of frame as key and heading as value
    fhd = {}
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            fhd['%04d'%int(split[0])] = split[-1]

    return fhd

def randerase_image(image, erase_ratio, size_min=0.02, size_max=0.4, ratio_min=0.3, ratio_max=1/0.3, val_min=0, val_max=255):
    """ Randomly erase a rectangular part of the given image in order to augment data
    https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py"""
    re_image = image.copy()
    h, w = re_image.shape
    er = random.random() # a float [0.0, 1.0)
    if er > erase_ratio:
        return None

    while True:
        size = np.random.uniform(size_min, size_max) * h * w
        ratio = np.random.uniform(ratio_min, ratio_max)
        width = int(np.sqrt(size / ratio))
        height = int(np.sqrt(size * ratio))
        left = np.random.randint(0, w)
        top = np.random.randint(0, h)
        if (left + width <= w and top + height <= h):
            break
    color = np.random.uniform(val_min, val_max)
    re_image[top:top+height, left:left+width] = color
    return re_image

def calculate_mean(images):
    # If adding additional channel with heading/cell identification, following lines can be problematic, watch out!
    depth = images[0].shape[-1]
    if (depth == image_size): #for grayscale images, .shape() only returns width and height
        N = 0
        mean = np.zeros((image_size,image_size))
        for img in images:
            mean[:, :] += img[:, :]
            N += 1
        mean /= N

    elif (depth == 3):
        N = 0
        mean = np.zeros((images[0].size[0], images[0].size[0], 3))
        for img in images:
            mean[:, :, 0] += img[:, :, 0]
            mean[:, :, 1] += img[:, :, 1]
            mean[:, :, 2] += img[:, :, 2]
            N += 1
        mean /= N
    else:
        # print(images.shape)
        # print("*** Check image shape")
        return None

    np.save(DATA + 'lstm_mean135k.npy', mean)
    print("*** Done. Returning mean.")
    return mean


def add_cell_channel(cell_frame_dict = None, rndUnderRepSubset = None , cellOutput = None, headOuput=None ):
    notNewImages = OrderedDict()
    newImages = OrderedDict()
    allImages = []
    frame_heading_dict = getFrameHeadingDict()
    frame_cell_dict = getFrameCellDict()

    def processFrame(frame):
        print( "Processing frame " + str(frameNum) + " / " + str(len(cell_frame_dict) * images_per_cell) + "     (Frame number: " + frame + ")")
        image = cv2.imread(DATA +'frames/moreframes/frame' + frame + '.jpg')
        image = resizeAndCrop(image)
        allImages.append(image)
        return image

    #Processing the frames into numpy images. One that is just getting the images according to the frame and the other
    #That is getting the image plus a grey rectangle
    frameNum = 1
    for cell in cell_frame_dict.keys():
        notNewImages[cell] = []
        whichFrame = 0
        for frame in cell_frame_dict[cell]:
            notNewImages[cell].append(processFrame(frame))
            whichFrame += 1
            frameNum += 1

    for cell in rndUnderRepSubset.keys():
        newImages[cell]= []
        whichFrame = 0
        for frame in rndUnderRepSubset[cell]:
            img = processFrame(frame)
            img = randerase_image(img, 1)
            newImages[cell].append(img)
            whichFrame += 1
            frameNum += 1

    #Merging the dictionaries so cell_frame_dict with rndUnderRepSubset, which only contain cell: ["frame", ...] format and
    #notNewImages with newImages, which contain cell: [image, ...] format

    for key in rndUnderRepSubset.keys(): #DATA in rndUnderRepSubset ----> cell_frame_dict
        for frame in rndUnderRepSubset[key]:
            cell_frame_dict[key].append(frame)

    for key in newImages.keys(): #DATA in newImages ----> notNewImages
        for imgs in newImages[key]:
            notNewImages[key].append(imgs)

    #Creating a tuple of frame with its corresponding image within each cell, so {cell: [("frame", image), ...]}
    #And sorting it according to the frame number

    for key in cell_frame_dict.keys(): #DATA in notNewImages ----> cell_frame_dict
        whichFrame = 0
        for frame in cell_frame_dict[key]:
            cell_frame_dict[key][whichFrame] = (int(frame), notNewImages[key][whichFrame])
            whichFrame += 1
        cell_frame_dict[key] = sorted(cell_frame_dict[key],key=lambda x: x[0])

    #Creating an array of images called train_IMG and an array of hot label either for cellOuput or cellInput
    train_IMG = []
    hotLabelcellOutput = []
    hotLabelHeadOutput = []

    for cell in cell_frame_dict.keys():
        train_IMG = cell_frame_dict[cell][1]
        frame = '%04d'% cell_frame_dict[cell][0]
        if cellOutput == True:
            hotLabelcellOutput.append(getOneHotLabel(int(frame_cell_dict[frame]), numCells))
        if headOuput == True:
            hotLabelHeadOutput.append(getOneHotLabel(int(frame_heading_dict[frame]) // 45, 8))


    #Calculating the mean
    mean = calculate_mean(train_IMG)

    #Suntracting the mean of images and normalizing each image
    for i in train_IMG:
        image = train_IMG[i]
        image = image - mean
        image /= 255
        image = np.squeeze(image)
        train_IMG[i] = image

    #ONLY USED FOR DOUBLE FEATURE IN LSTM or cellinput/headInput
    # whichImage = 0
    # for cell in cell_frame_dict.keys():
    #     frame = cell_frame_dict[cell][0]
    #     if headInput == True:
    #         head = int(frame_heading_dict[frame])
    #         head_arr = head * np.ones((train_IMG[whichImage].shape[0], train_IMG[whichImage].shape[1], 1))
    #         train_IMG[whichImage] = np.concatenate((np.expand_dims(train_IMG[whichImage], axis=-1), head_arr), axis=-1)
    #     if cellInput == True:
    #         cell = int(frame_cell_dict[frame])
    #         cell_arr = cell * np.ones((train_IMG[whichImage].shape[0], train_IMG[whichImage].shape[1], 1))
    #         train_IMG[whichImage] = np.concatenate((np.expand_dims(train_IMG[whichImage], axis=-1), cell_arr), axis=-1)
    #     whichImage +=1



    train_IMG = np.asarray(train_IMG)
    hotLabelHeadOutput = np.asarray(hotLabelHeadOutput)
    np.save(DATA + "Img", train_IMG)
    np.save(DATA + "head", hotLabelHeadOutput)



if __name__ == '__main__':
    # cell_counts, cell_frame_dict = getCellCounts()
    # cell_heading_counts = getHeadingRep(cell_counts)
    # cullOverRepped(cell_counts, cell_frame_dict, cell_heading_counts)
    # addUnderRepped(cell_counts, cell_frame_dict, cell_heading_counts)

    cell_frame_dict = np.load(DATA+ 'cell_origFrames.npy',allow_pickle='TRUE').item()
    rndUnderRepSubset = np.load(DATA + 'cell_newFrames.npy', allow_pickle='TRUE').item()
    ################################################################
    #Selecting the SAMPLE
    wantedCells = ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                   '35', '36', '37', '38', '39', '40', '41', '42', '43']
    frame_dict = OrderedDict()
    newFrames = OrderedDict()
    for cell in wantedCells:
        frame_dict[cell] = cell_frame_dict[cell]
        if(len(rndUnderRepSubset[cell]) > 0):
            newFrames[cell]= rndUnderRepSubset[cell]

    add_cell_channel(frame_dict ,newFrames , cellInput= None, headingInput=True)
    ################################################################



    # images = np.load(DATA + "lstm_Img_Cell_Input.npy")
    # images = images[:, :, :, 0]
    # # images = images.reshape(12500, 100, 100, 1)
    # images = images.reshape(25, 500, 100, 100, 1)
    # cell = 4
    # #for i in range(100, 12501, 100):
    #
    # for frame in  images[cell]:
    #     cv2.imshow("window", frame)
    #     cv2.waitKey(30)
    # cv2.destroyAllWindows()
    #
    # for i in range(0, 500, 40):
    #     print("start",str(i), "end", str(i+39))
    #     ten = np.concatenate((images[cell][i],images[cell][i+1], images[cell][i+2], images[cell][i+3], images[cell][i+4],
    #                       images[cell][i+5], images[cell][i+6], images[cell][i+7], images[cell][i+8], images[cell][i+9]),
    #                          axis=1)
    #     twen = np.concatenate((images[cell][i +10],images[cell][i+11], images[cell][i+12], images[cell][i+13], images[cell][i+14],
    #                       images[cell][i+15], images[cell][i+16], images[cell][i+17], images[cell][i+18], images[cell][i+19]), axis=1)
    #     thrt = np.concatenate((images[cell][i +20],images[cell][i+21], images[cell][i+22], images[cell][i+23], images[cell][i+24],
    #                       images[cell][i+25], images[cell][i+26], images[cell][i+27], images[cell][i+28], images[cell][i+29]), axis=1)
    #     frty = np.concatenate((images[cell][i +30],images[cell][i+31], images[cell][i+32], images[cell][i+33], images[cell][i+34],
    #                       images[cell][i+35], images[cell][i+36], images[cell][i+37], images[cell][i+38], images[cell][i+39]),
    #                          axis=1)
    #
    #
    #
    #     # tenImgs = np.concatenate((images[frame], images[frame + 1], images[frame + 2], images[frame + 3], images[frame + 4],
    #     #      images[frame + 5], images[frame + 6], images[frame + 7], images[frame + 8], images[frame + 9]),
    #     #     axis=1)
    #     # anotherTEN = np.concatenate((images[frame + 10], images[frame + 11], images[frame + 12],
    #     #                              images[frame + 13], images[frame + 14],
    #     #                              images[frame + 15], images[frame + 16], images[frame + 17],
    #     #                              images[frame + 18], images[frame + 19]), axis=1)
    #     img = np.concatenate((ten, twen, thrt, frty), axis=0)
    #     cv2.imshow('Window',img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()



