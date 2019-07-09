# This program divides the image data into two parts, overrepped, with images corresponding to cells that have too much
# and underrepped, with images corresponding to cells that have too little. To cull the overrepped data, for each
# overrepped cell we iteratively removed a random image with the most represented heading until we had the desired
# number of images. This is the new and immproved version of olin_inputs.py
# Authors: Avik Bosshardt, Angel Sylvester, Maddie AlQatami

import olin_factory as factory
import cv2
import numpy as np
import os
import random
from datetime import datetime
from olin_cnn import loading_bar

numCells = 153
image_size = 100
images_per_cell = 500

def getCellCounts():
    with open(factory.paths.cell_data_path,'r') as masterlist:
        lines = masterlist.readlines()
        cell_counts = dict()
        cell_frame_dict = dict()
        for i in range(numCells):
            cell_frame_dict[str(i)] = []

        for line in lines:
            splitline = line.split()
            if splitline[1] not in cell_counts.keys():
                cell_counts[splitline[1]] = 1
            else:
                cell_counts[splitline[1]] += 1

            cell_frame_dict[splitline[1]].append('%04d'%int(splitline[0]))

    return cell_counts, cell_frame_dict

def getFrameCellDict():
    frame_cell_dict = {}
    with open(factory.paths.cell_data_path,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            frame_cell_dict['%04d'%int(split[0])] = split[1]
    return frame_cell_dict

def getHeadingRep():
    cellHeadingCounts = dict()
    for i in range(numCells):
        cellHeadingCounts[str(i)] = [['0',0],['45',0],['90',0],['135',0],['180',0],['225',0],['270',0],['315',0]]
    with open(factory.paths.cell_data_path,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            for pair in cellHeadingCounts[str(split[1])]:
                if pair[0] == split[-1]:
                    pair[1] += 1
                    break

    return  cellHeadingCounts

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

    with open(factory.paths.cell_data_path,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            heading_frame_dict[split[-1]].append('%04d'%int(split[0]))

    return heading_frame_dict

def getFrameHeadingDict():
    fhd = {}
    with open(factory.paths.cell_data_path,'r') as masterlist:
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
    remove = []
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

    np.save('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_overreppedFrames.npy',overreppedFrames)
    return overreppedFrames

def addUnderRepped():
    cell_counts, cell_frame_dict = getCellCounts()
    cell_heading_counts = getHeadingRep()
    underRepList, over = getUnderOverRep(cell_counts)
    heading_frame_dict = getHeadingFrameDict()
    remove = []
    underreppedFrames = []
    i = 1
    for cell in underRepList:
        print('Cell '+ str(i) + " of " + str(len(underRepList)),cell)
        i+=1

        for frame in cell_frame_dict[cell]:
            underreppedFrames.append(frame)
        while cell_counts[cell] < images_per_cell:
            headingList = sorted(cell_heading_counts[cell],key= lambda x: x[1])
            smallestHeading = headingList[0][0]
            headingList[0][1] = headingList[0][1] + 1
            potentialAdditions = []
            for frame in heading_frame_dict[smallestHeading]:
                if frame in cell_frame_dict[cell]:
                    potentialAdditions.append(frame)
            if len(potentialAdditions) == 0:
                print(cell, 'has very little data')
                continue
            toBeAdded = random.choice(potentialAdditions)
            underreppedFrames.append(toBeAdded)
            cell_frame_dict[cell].append(toBeAdded)
            cell_counts[cell] += 1

    print(len(underreppedFrames))
    np.save('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_underreppedFrames.npy',underreppedFrames)
    return underreppedFrames

def resizeAndCrop(image):
    # Original size 341x256 cropped to 224x224
    # smaller size 170x128 cropped to 100x100

    cropped_image = cv2.resize(image, (image_size,image_size))
    # image = cv2.resize(image,(170,128))
    # x = random.randrange(0,70)
    # y = random.randrange(0,28)
    #
    #cropped_image = image[y:y+100, x:x+100]
    cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

    return cropped_image

def add_cell_channel(overRepped=None, underRepped=None):
    frame_cell_dict = getFrameCellDict()
    frame_heading_dict = getFrameHeadingDict()
    training_data = []
    allImages = []

    def processFrame(frame):
        print "Processing frame " + str(frameNum) + " / " + str(
            len(overRepped) + len(underRepped)) + "     (Frame number: " + frame + ")"

        image = cv2.imread(factory.paths.train_data_dir + '/frame' + frame + '.jpg')
        image = resizeAndCrop(image)
        allImages.append(image)

        cell = int(frame_cell_dict[frame])
        cell_arr = cell * np.ones((image.shape[0], image.shape[1], 1))
        training_data.append([image,
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

    for i in range(len(training_data)):
        if i > len(underRepped)-1:
            frame = overRepped[i-len(underRepped)]
        else:
            frame = underRepped[i]
        loading_bar(i,len(training_data))
        image = training_data[i][0]
        image = image - mean
        image = np.squeeze(image)
        re_image = randerase_image(image, 1)
        if re_image is None:
            re_image = image

        cell = int(frame_cell_dict[frame])
        cell_arr = cell*np.ones((image.shape[0], image.shape[1], 1))
        training_data[i][0] = np.concatenate((np.expand_dims(re_image,axis=-1),cell_arr),axis=-1)

    np.save(
        '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_' + str(
            image_size) + '_' + str(images_per_cell) + 'withCellInput.npy', training_data)

    print('Done!')
    return training_data

def calculate_mean(images):
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
        mean = np.zeros((factory.image.size, factory.image.size, 3))
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
        print "Processing frame " + str(frameNum) + " / " + str(len(overRepped) + len(underRepped))+ "     (Frame number: " + frame + ")"


        image = cv2.imread(factory.paths.train_data_dir + '/frame' + frame + '.jpg')
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

    np.save('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_'+str(image_size)+'_'+str(images_per_cell)+'.npy',training_data)

    print('Done!')
    return training_data

if __name__ == '__main__':
    # getTrainingData(underRepped=np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_underreppedFrames.npy'),
    # overRepped=np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_overreppedFrames.npy'))
    # getTrainingData()

    # over = np.load(
    #    '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_overreppedFrames.npy')
    #under = np.load(
    #   '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_underreppedFrames.npy')
    # arr = []
    # for cell in getCellCounts()[0].keys():
    #
    #     arr.append(int(cell))
    # arr.sort()
    # print(arr)
    # for i in range(len(arr)-1):
    #     if arr[i+1] - arr[i] != 1:
    #         print(i,arr[i])
# for i in range(95837):
    #     if '%04d'%i not in over and '%04d'%i not in under:
    #         print i
    # data = np.load(
    #     "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_100_gray_norm_randerase.npy")
    # for image in data[:1000]:
    #     cv2.imshow("image", image[0])
    #     cv2.waitKey(0)
    add_cell_channel(underRepped=np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_underreppedFrames.npy'),
    overRepped=np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/newdata_overreppedFrames.npy'))
