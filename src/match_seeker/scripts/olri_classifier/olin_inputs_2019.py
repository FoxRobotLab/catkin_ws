
# This program divides the image data into two parts, overrepped, with images corresponding to cells that have too much
# and underrepped, with images corresponding to cells that have too little. To cull the overrepped data, for each
# overrepped cell we iteratively removed a random image with the most represented heading until we had the desired
# number of images. This is the new and immproved version of olin_inputs.py
# Authors: Avik Bosshardt, Angel Sylvester, Maddie AlQatami

import cv2
import numpy as np
import os
import random
from datetime import datetime
from olin_cnn import loading_bar
from paths import pathToClassifier2019 

numCells = 50 #ORIG 271 
image_size = 100
images_per_cell = 500
master_cell_loc_frame_id = pathToClassifier2019 + '/frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt' #'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'

def getCellCounts():
    # Counts number of frames per cell, also returns dictionary with list of all frames associated with each cell
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        cell_counts = dict()
        cell_frame_dict = dict()
        for i in range(271): #ORIG range(numCells) ALSO COULD BE WASTING SPACE
            cell_frame_dict[str(i)] = []

        for line in lines:
            splitline = line.split()
            if splitline[1] not in cell_counts.keys():
		if len(cell_counts) >= numCells: #DT and len(cell_counts) is not numbCells
		    continue #DT
                cell_counts[splitline[1]] = 1
            else:
                cell_counts[splitline[1]] += 1

            cell_frame_dict[splitline[1]].append('%04d'%int(splitline[0]))
	    

    return cell_counts, cell_frame_dict

def getFrameCellDict():
    # Returns dict with cell corresponding to each frame
    frame_cell_dict = {}
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            frame_cell_dict['%04d'%int(split[0])] = split[1]
    return frame_cell_dict

def getHeadingRep():
    # Returns dict with cell --> counts of num frames taken at each heading in that cell
    cellHeadingCounts = dict()
    for i in range(271): #ORIG range(numCells) 
        cellHeadingCounts[str(i)] = [['0',0],['45',0],['90',0],['135',0],['180',0],['225',0],['270',0],['315',0]]
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            for pair in cellHeadingCounts[str(split[1])]:
                if pair[0] == split[-1]:
                    pair[1] += 1
                    break

    return cellHeadingCounts

def getUnderOverRep(cell_counts):
    underRep = []
    overRep = []

    for key in cell_counts.keys():
        if int(cell_counts[key]) <= images_per_cell:
            underRep.append(key)
        else:
            overRep.append(key)

    return underRep, overRep

def getHeadingFrameDict():
    heading_frame_dict = {'0':[],'45':[],'90':[],'135':[],'180':[],'225':[],'270':[],'315':[]}

    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            heading_frame_dict[split[-1]].append('%04d'%int(split[0]))

    return heading_frame_dict

def getFrameHeadingDict():
    fhd = {}
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            fhd['%04d'%int(split[0])] = split[-1]

    return fhd

def getOneHotLabel(number,size):
    onehot = [0] * size
    onehot[number] = 1
    return onehot

def cullOverRepped():
    cell_counts, cell_frame_dict = getCellCounts()
    cell_heading_counts = getHeadingRep()
    under, overRepList = getUnderOverRep(cell_counts)
    heading_frame_dict = getHeadingFrameDict()
    frame_to_heading = getFrameHeadingDict()
    overreppedFrames = []
    i = 1
    for cell in overRepList:
        print('Cell '+ str(i) + " of " + str(len(overRepList)))
        i+=1

        for frame in cell_frame_dict[cell]:
            overreppedFrames.append(frame)
        while cell_counts[cell] > images_per_cell:
            headingList = sorted(cell_heading_counts[cell],key= lambda x: x[1])
            largestHeading = headingList[-1][0]
            headingList[-1][1] = headingList[-1][1] -1

            potentialCulls = []
            for frame in heading_frame_dict[largestHeading]:
                if frame in cell_frame_dict[cell]:
                    potentialCulls.append(frame)
            toBeRemoved = potentialCulls[random.randint(0,len(potentialCulls)-1)]
            overreppedFrames.remove(toBeRemoved)

            cell_frame_dict[cell].remove(toBeRemoved)
            cell_counts[cell] -= 1

    print(len(overreppedFrames))
    # for frame in remove:
    #     try:
    #         overreppedFrames.remove(frame)
    #     except ValueError:
    #         print(frame)

    np.save(pathToClassifier2019 + '/sampleTest_overreppedFrames25k.npy', overreppedFrames)
    #ORIG np.save(pathToClassifier2019 +  '/newdata_overreppedFrames95k.npy',overreppedFrames)
    return overreppedFrames

def addUnderRepped():
    cell_counts, cell_frame_dict = getCellCounts()
    cell_heading_counts = getHeadingRep()
    underRepList, over = getUnderOverRep(cell_counts)
    heading_frame_dict = getHeadingFrameDict()
    underreppedFrames = []
    rndUnderRepSubset = []
    i = 1
    for cell in underRepList:
        print('Cell '+ str(i) + " of " + str(len(underRepList)),cell)
        i+=1

        for frame in cell_frame_dict[cell]:
            underreppedFrames.append(frame)
        while cell_counts[cell] < images_per_cell:
            headingList = sorted(cell_heading_counts[cell],key= lambda x: x[1])
	    h = 0 #PC
	    while(headingList[h][1] == 0):#PC
	        h+=1 #PC
            smallestHeading = headingList[h][0] #BORIG smallestHeading = headingList[0][0]
            headingList[h][1] = headingList[h][1] + 1 #ORIG headingList[0][1] = headingList[0][1] + 1
            potentialAdditions = []
            for frame in heading_frame_dict[smallestHeading]: 
                if frame in cell_frame_dict[cell]: 
                    potentialAdditions.append(frame) 
            if len(potentialAdditions) == 0:#UNNECESSARY?
                print(cell, 'has very little data')
                continue
            toBeAdded = random.choice(potentialAdditions)
            rndUnderRepSubset.append(toBeAdded)
            cell_frame_dict[cell].append(toBeAdded)
            cell_counts[cell] += 1

    
    #PC Also save rndUnderRepSubset as a npy file
    np.save(pathToClassifier2019+ '/sampleTest_underreppedFrames25k.npy', underreppedFrames)
    np.save(pathToClassifier2019+ '/sampleTest_rndUnderRepSubsetFrames25k.npy', rndUnderRepSubset)
    #ORIG np.save(pathToClassifier2019 +'/newdata_underreppedFrames95k.npy',underreppedFrames)
    return underreppedFrames, rndUnderRepSubset

def resizeAndCrop(image):
    # 341x256 cropped to 224x224  -OR-
    # Smaller size 170x128 cropped to 100x100
    if image is None:
	print("No Image")
    else:
    	cropped_image = cv2.resize(image, (image_size,image_size))

    ### Uncomment to crop square and preserve aspect ratio ###
    # image = cv2.resize(image,(170,128))
    # x = random.randrange(0,70)
    # y = random.randrange(0,28)
    #
    # cropped_image = image[y:y+100, x:x+100]
    	cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

    	return cropped_image

def add_cell_channel(underRepped=None, randomUnderRepSubset=None, overRepped=None):
    frame_cell_dict = getFrameCellDict()
    frame_heading_dict = getFrameHeadingDict()
    training_data = []
    allImages = []

    if underRepped is None:
        underRepped, randomUnderRepSubset = addUnderRepped()
    if overRepped is None:
        overRepped = cullOverRepped()



    def processFrame(frame):
        print( "Processing frame " + str(frameNum) + " / " + str(
            len(overRepped) + len(underRepped) + len(randomUnderRepSubset)) + "     (Frame number: " + frame + ")")
        image = cv2.imread(pathToClassifier2019 +'/frames/moreframes/frame' + frame + '.jpg')
        image = resizeAndCrop(image)
        allImages.append(image)
	return image

        
       

    frameNum = 1
    for frame in underRepped:
        img= processFrame(frame)
	training_data.append([img, getOneHotLabel(int(frame_heading_dict[frame]) // 45, 8)])
        frameNum += 1
    for frame in randomUnderRepSubset:
	img = processFrame(frame)
	new_img = randerase_image(img, 1)
	training_data.append([new_img, getOneHotLabel(int(frame_heading_dict[frame]) // 45, 8)])
	frameNum +=1


    for frame in overRepped:
        img = processFrame(frame)
	training_data.append([img, getOneHotLabel(int(frame_heading_dict[frame]) // 45, 8)])
        frameNum += 1



    mean = calculate_mean(allImages)
    loading_bar(frameNum, len(overRepped) + len(underRepped) + len(randomUnderRepSubset), 150)

    
        #if i > len(underRepped)-1:
            #frame = overRepped[i-len(underRepped)]
        #else:
            #frame = underRepped[i]
    for i in range(len(training_data)):
	if (i < len(underRepped)):
	    frame = underRepped[i]
	elif (i < len(underRepped) + len(randomUnderRepSubset)):
	    frame = randomUnderRepSubset[i-len(underRepped)]
	else:
	    frame = overRepped[i - (len(underRepped)+len(randomUnderRepSubset))] 
        loading_bar(i,len(training_data))
        image = training_data[i][0]
        image = image - mean
        image = np.squeeze(image)
        cell = int(frame_cell_dict[frame])
        cell_arr = cell*np.ones((image.shape[0], image.shape[1], 1))
        training_data[i][0] = np.concatenate((np.expand_dims(image, axis=-1), cell_arr), axis=-1)

        # heading = (int(frame_heading_dict[frame])) // 45
        # heading_arr = heading*np.ones((image.shape[0], image.shape[1], 1))
        # training_data[i][0] = np.concatenate((np.expand_dims(image,axis=-1),heading_arr),axis=-1)

    #np.save(pathToClassifier2019 + '/NEWTRAININGDATA_' + str(
            #image_size) + '_' + str(images_per_cell) + 'withCellInput95k.npy', training_data)
    np.save(pathToClassifier2019 + '/SAMPLETRAININGDATA_100_500_withCellInput25K.py', training_data)

    print('Done!')
    return training_data

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
    ### IF USING NEW IMAGE SET, BE SURE TO SAVE MEAN!!
    #np.save('TRAININGDATA_100_500_mean95k.npy',mean)
    np.save('SAMPLETRAINING_100_500_mean25k.npy', mean)
    print("*** Done. Returning mean.")
    return mean

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

def getTrainingData(overRepped=None,underRepped=None):
    frame_cell_dict = getFrameCellDict()
    frame_heading_dict = getFrameHeadingDict()
    training_data = []
    allImages = []

    def processFrame(frame):
        print ("Processing frame " + str(frameNum) + " / " + str(len(overRepped) + len(underRepped))+ "     (Frame number: " + frame + ")")

        image = cv2.imread(pathToClassifier2019 +'/frames/moreframes/frame' + frame + '.jpg')
        image = resizeAndCrop(image)
        allImages.append(image)

        training_data.append([np.array(image), getOneHotLabel(int(frame_cell_dict[frame]), numCells),
                              getOneHotLabel(int(frame_heading_dict[frame]) // 45, 8)])


    frameNum = 1
    if underRepped is not None:
        for frame in underRepped:
            processFrame(frame)
            frameNum += 1
    else:
        underRepped = addUnderRepped()
        for frame in underRepped:
            processFrame(frame)
            frameNum += 1

    if overRepped is not None:
        for frame in overRepped:
            processFrame(frame)
            frameNum += 1
    else:
        overRepped = cullOverRepped()
        for frame in overRepped:
            processFrame(frame)
            frameNum += 1

    mean = calculate_mean(allImages)
    loading_bar(frameNum, len(overRepped) + len(underRepped), 150)

    ##############################################################################################
    ### Uncomment to preprocess data with randerasing and normalization (thru mean subtraction)###
    ### Don't forget to change resizeAndCrop() to convert to gray/alter the resizing method    ###
    ### (e.g. preserve aspect ratio vs squish  image)                                          ###
    ##############################################################################################

    # for i in range(len(training_data)):
    #     loading_bar(i,len(training_data))
    #     image = training_data[i][0]
    #     image = image - mean
    #     image = np.squeeze(image)
    #     re_image = randerase_image(image, 1)
    #     if re_image is None:
    #         re_image = image
    #
    #     training_data[i][0] = re_image

    np.save(pathToClassifier2019 + '/NEWTRAININGDATA_'+str(image_size)+'_'+str(images_per_cell)+'95k.npy',training_data)

    print('Done!')
    return training_data

if __name__ == '__main__':
    add_cell_channel(underRepped= np.load(pathToClassifier2019 + 'sampleTest_underreppedFrames25k.npy'), randomUnderRepSubset = np.load(pathToClassifier2019 + 'sampleTest_rndUnderRepSubsetFrames25k.npy'), overRepped = np.load(pathToClassifier2019 + 'sampleTest_overreppedFrames25k.npy'))
    #add_cell_channel()
    #add_cell_channel(underRepped=np.load(pathToClassifier2019 + '/newdata_underreppedFrames95k.npy'),
                      #overRepped=np.load(pathToClassifier2019 + '/newdata_overreppedFrames95k.npy'))
