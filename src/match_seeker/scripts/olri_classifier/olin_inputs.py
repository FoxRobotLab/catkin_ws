import os
import cv2
import numpy as np
import random
import olin_factory as factory
import time

################## Preprocessing of data ##################
def read_cell_data(cell_file=factory.paths.cell_data_path):
    """
    Read the cell data file created by locsToCells.py and converts them to a
    dictionary. Referred to locsToCells.py
    :param cell_infile: a txt file that contains <frame#> <cell#> <yaw> each line
    :return: a dictionary that has cell number as a key and a list containing the
    corresponding cell number and yaw as a value
    """
    cell_file = open(cell_file, 'r')
    cell_dict = dict()
    for line in cell_file:
        if line == "" or line.isspace() or line[0] == '#':
            continue
        parts = line.split()
        frameNum = parts[0]
        cell = int(parts[1])
        yaw = float(parts[-1])
        cell_dict[frameNum] = [cell, yaw]
    cell_file.close()
    print("*** Reading cell data and returning as a dictionary...")
    return cell_dict
def _extractNum(fileString):
    """finds sequence of digits"""
    numStr = ""
    foundDigits = False
    for c in fileString:
        if c in '0123456789':
            foundDigits = True
            numStr += c
        elif foundDigits == True:
            break
    if numStr != "":
        return int(numStr)
    else:
        print("*** ERROR: no number in ", fileString)


def calculate_mean(images):
    depth = images[0].shape[-1]
    if (depth == factory.image.size):
        N = 0
        mean = np.zeros((factory.image.size, factory.image.size))
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
        print(images.shape)
        print("*** Check image shape")
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

def warp_image(image):
    """ Warp perspective of the given image in order to augment data """
    h, w, d = image.shape
    cv2.imshow("Original Image",image)
    warp = cv2.getPerspectiveTransform(
        src=np.float32([ [0,0], [0,w], [h*0.85,0], [h*0.85, w*0.85] ]),
        dst=np.float32([ [0,0], [0,w], [h,0], [h,w] ])
    )
    warped_image = cv2.warpPerspective(image, warp, (w, h))
    cv2.imshow("Warped Image", warped_image)
    k = cv2.waitKey(0)
    ch = chr(k & 0xFF)
    if ch == "q": cv2.destroyAllWindows()
    return warped_image

def augment_data():
    pass

def get_np_train_images_and_labels(train_data):
    train_images = np.array([i[0] for i in train_data]) \
        .reshape(-1, factory.image.size, factory.image.size, factory.image.depth)
    train_labels = np.array([i[1] for i in train_data])
    return train_images, train_labels

def create_train_data(extension=".jpg", ascolor=False, normalize=True, exclude_ratio=None, exclude_number=None, randerase_ratio=None, max_images_per_category=None):
    """
    Makes and saves training data
    https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
    :param extension:
    :param train_data_name:
    :param ascolor: leave the images colored if True
    :param normalize: execute mean subtraction if True
    :param exclude_ratio: a float value that tells the ratio of how much data to keep from the training data set classes that do not have enough data
    :return: An array containing training data with the format of [np.array(image), np.array(label)]
    """
    print("*** Processing train data... this may take a few minutes...")

    frame_to_cellyaw_dict = read_cell_data(factory.paths.cell_data_path) #int float
    training_data = []
    train_data_tag = ""
    cell_counts = [0] * factory.cell.num_max_cells # number of each cell count with array index as cell number and value as cell count

    cell_to_frames_dict = dict()

    for filename in os.listdir(factory.paths.train_data_dir):
        if (filename.endswith(extension)):
            frame_num = str(int(_extractNum(filename)))
            cell_num, yaw = frame_to_cellyaw_dict[frame_num]

            cell_counts[cell_num] += 1

            path = os.path.join(factory.paths.train_data_dir, filename)
            img = cv2.imread(filename=path)
            # Prevent the network from learning by color
            if (not ascolor):
                if (not "-gray" in train_data_tag): train_data_tag += "-gray"
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_img = cv2.resize(gray_img, (factory.image.size, factory.image.size))
            else:
                if (not "-color" in train_data_tag): train_data_tag += "-color"
                resized_img = cv2.resize(img, (factory.image.size, factory.image.size))


            if (randerase_ratio is not None):
                if (not "-re" in train_data_tag): train_data_tag += "-re" + str(randerase_ratio)
                re_image = randerase_image(resized_img, randerase_ratio)
                if (re_image is not None):
                    if (cell_num in cell_to_frames_dict.keys()):
                        cell_to_frames_dict[cell_num].append(re_image)
                    else:
                        cell_to_frames_dict[cell_num] = [re_image]

                cell_counts[cell_num] += 1

            if not (normalize):
                training_data.append([np.array(resized_img), cell_num]) # TODO: Modify labeling when label dict is changed (array -> num)
            else:
                if (cell_num in cell_to_frames_dict.keys()):
                    cell_to_frames_dict[cell_num].append(resized_img)
                else:
                    cell_to_frames_dict[cell_num] = [resized_img]

    print("*** Cell Count before exclusion:\n", cell_counts)
    # exit() # uncomment to see cell counts

    ### Exclude data in cells that have less than exclude_ratio of maximum cell count
    exclude_labels = [] # cell_nums == labels
    if (exclude_ratio is not None):
        train_data_tag += "-er"+str(exclude_ratio)
        exclude_num = max(cell_counts) * exclude_ratio
        for cell_num in range(len(cell_counts)):
            if (cell_counts[cell_num] < exclude_num):
                exclude_labels.append(cell_num)
    elif (exclude_number is not None):
        train_data_tag += "-en" + str(exclude_number)
        exclude_num = exclude_number
        for cell_num in range(len(cell_counts)):
            if (cell_counts[cell_num] < exclude_num):
                exclude_labels.append(cell_num)
    if (len(exclude_labels) != 0):
        print("*** Excluding cells that lack enough data (< {}): ".format(exclude_num), exclude_labels)
        for exclude_cell in exclude_labels:
            if (not exclude_cell in cell_to_frames_dict.keys()):
                continue
            del cell_to_frames_dict[exclude_cell]
            cell_counts[exclude_cell] = 0

        print("*** Cell Labels with len {} contain these unique labels".format(len(cell_to_frames_dict.keys())),set(cell_to_frames_dict.keys()))


    ### Randomly pick out only some data if too much data for a cell
    if (max_images_per_category is not None):
        print("*** Limiting each cell to have maximum of {} data".format(max_images_per_category))
        train_data_tag += "-max" + str(max_images_per_category)
        ### Count cells
        toomuch_labels = []
        for cell_num in range(len(cell_counts)):
            if (cell_counts[cell_num] > max_images_per_category):
                toomuch_labels.append(cell_num)
        print("*** Cell with too much data: ", toomuch_labels)

        ### Remove data
        for cell in toomuch_labels:
            np.random.shuffle(cell_to_frames_dict[cell])
            cell_to_frames_dict[cell] = cell_to_frames_dict[cell][0:max_images_per_category]
            cell_counts[cell] = max_images_per_category

        print("*** Cell counts after limiting max :", cell_counts)


    to_one_hot_dict = dict()
    counter = 0
    for unique_label in set(cell_to_frames_dict.keys()):
        to_one_hot_dict[unique_label] = counter
        counter += 1

    images = []
    one_hot_labels = []
    for label in cell_to_frames_dict.keys():
        one_hot = [0] * len(set(cell_to_frames_dict.keys()))
        one_hot[to_one_hot_dict[label]] = 1
        for image in cell_to_frames_dict[label]:
            one_hot_labels.append(one_hot)
            images.append(image)
    print("*** int to one hot dict",to_one_hot_dict)



    mdyHM_str = time.strftime("%m%d%y%H%M")  # e.g. 0706181410 (mmddyyHHMM)


    ### Calculate and Subtract mean from all images
    if (normalize):
        train_data_tag += "-submean"
        mean = calculate_mean(images)
        np.save(mdyHM_str + "train_mean" + train_data_tag + ".npy", mean)
        print("*** Saved mean as...{}".format("train_mean" + train_data_tag + ".npy"))
        print("*** Mean subtraction normalization with the mean: ")
        print(mean)
        for i in range(len(images)):
            norm_image = images[i] - mean
            norm_image = np.squeeze(norm_image)
            training_data.append([np.array(norm_image), np.array(one_hot_labels[i])])

        print("*** Image : \n", images[-1])
        print("*** Mean Sub Image: \n", norm_image)




    ### Save the data and Report the progress
    random.shuffle(training_data)  # Makes sure the frames are not in order (which could cause training to go bad...)
    np.save(mdyHM_str + "train_data" + train_data_tag + ".npy", training_data)
    np.save(mdyHM_str + "train_cellcounts" + train_data_tag + ".npy", cell_counts)
    np.save(mdyHM_str + "train_onehotdict" +train_data_tag + ".npy", to_one_hot_dict)
    print("*** Final cell Count:\n", cell_counts)
    print("*** Done. Saved train data as {}".format("train_data" + train_data_tag + ".npy"))
    print("*** Done. Saved train class count array as {}".format("train_cellcounts" + train_data_tag + ".npy"))
    print("*** Done. Saved train to one hot dict as {}".format("train_onehotdict" +train_data_tag + ".npy"))
    return training_data


def main():
    ### Calculate and Save Mean
    # image_files = glob.glob(factory.paths.train_data_dir+"*.jpg")
    # imgs = []
    # for f in image_files[:100]:
    #     img=cv2.imread(f)
    #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     resized_img = cv2.resize(gray_img, (80, 80))
    #     imgs.append(resized_img)
    # calculate_mean(images=imgs)

    ### Read Cell Data



    ### Test randerase_image
    # for image in os.listdir("frames/lessframes/"):
    #     if ("jpg" not in image):
    #         continue
    #     image = cv2.imread("frames/lessframes/"+image)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     image = cv2.resize(image, (100, 100))
    #     randerase_image(image)

    # warp_image(image)


    ### Create Train Data
    create_train_data(exclude_number=275, randerase_ratio=1.0, max_images_per_category=300)

if __name__ == "__main__":
    main()


"""
create_train_data()

072018 1130am
/home/macalester/tensorflow/bin/python /home/macalester/PycharmProjects/olri_classifier/olin_inputs.py
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count before exclusion:\n', [574, 458, 494, 272, 430, 492, 368, 356, 324, 334, 366, 510, 256, 254, 264, 472, 544, 646, 0, 162, 606, 296, 1256, 732, 776, 694, 768, 370, 930, 1398, 1298, 1358, 1344, 1288, 1706, 1554, 1228, 1458, 1166, 1992, 1354, 1476, 1000, 302, 658, 576, 306, 992, 666, 566, 606, 548, 616, 626, 646, 688, 550, 574, 516, 574, 618, 532, 668, 548, 480, 444, 930, 528, 836, 764, 880, 756, 1336, 844, 794, 776, 894, 820, 882, 1522, 1312, 1212, 902, 946, 900, 1000, 1164, 1074, 776, 652, 606, 544, 714, 678, 596, 262, 984, 272, 1162, 520, 780, 522, 1132, 1112, 672, 684, 752, 684, 612, 852, 810, 876, 804, 1166, 508, 502, 572, 1024, 1070, 450, 676, 554, 824, 254, 694, 968, 1182, 762, 796, 802, 750, 760, 972, 532, 1060, 872, 812, 726, 774, 344, 1766, 872, 734, 1158, 614, 550, 594, 604, 558, 590, 610, 532, 460])
('*** Excluding cells that lack enough data (< 275): ', [3, 12, 13, 14, 18, 19, 95, 97, 123])
('*** Cell Labels with len 144 contain these unique labels', set([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152]))
*** Limiting each cell to have maximum of 300 data
('*** Cell with too much data: ', [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152])
('*** Cell counts after limiting max :', [300, 300, 300, 0, 300, 300, 300, 300, 300, 300, 300, 300, 0, 0, 0, 300, 300, 300, 0, 0, 300, 296, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 0, 300, 0, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 0, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
('*** int to one hot dict', {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 15: 11, 16: 12, 17: 13, 20: 14, 21: 15, 22: 16, 23: 17, 24: 18, 25: 19, 26: 20, 27: 21, 28: 22, 29: 23, 30: 24, 31: 25, 32: 26, 33: 27, 34: 28, 35: 29, 36: 30, 37: 31, 38: 32, 39: 33, 40: 34, 41: 35, 42: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 66: 60, 67: 61, 68: 62, 69: 63, 70: 64, 71: 65, 72: 66, 73: 67, 74: 68, 75: 69, 76: 70, 77: 71, 78: 72, 79: 73, 80: 74, 81: 75, 82: 76, 83: 77, 84: 78, 85: 79, 86: 80, 87: 81, 88: 82, 89: 83, 90: 84, 91: 85, 92: 86, 93: 87, 94: 88, 96: 89, 98: 90, 99: 91, 100: 92, 101: 93, 102: 94, 103: 95, 104: 96, 105: 97, 106: 98, 107: 99, 108: 100, 109: 101, 110: 102, 111: 103, 112: 104, 113: 105, 114: 106, 115: 107, 116: 108, 117: 109, 118: 110, 119: 111, 120: 112, 121: 113, 122: 114, 124: 115, 125: 116, 126: 117, 127: 118, 128: 119, 129: 120, 130: 121, 131: 122, 132: 123, 133: 124, 134: 125, 135: 126, 136: 127, 137: 128, 138: 129, 139: 130, 140: 131, 141: 132, 142: 133, 143: 134, 144: 135, 145: 136, 146: 137, 147: 138, 148: 139, 149: 140, 150: 141, 151: 142, 152: 143})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-re1.0-en275-max300-submean.npy
*** Mean subtraction normalization with the mean: 
[[139.95782017 141.20099083 140.43867488 ... 127.77254838 125.67321048
  122.09577276]
 [139.93802667 140.56486712 140.9960876  ... 128.24756922 126.58044726
  124.09366608]
 [141.18779517 141.36709418 141.18916103 ... 128.97025188 128.27037226
  126.02782665]
 ...
 [114.79470321 116.43089638 115.09183721 ... 109.55090749 107.38255857
  103.00710714]
 [113.43096583 115.97684971 116.4367997  ... 108.97731271 105.47284471
  101.36667747]
 [111.82204371 115.23474396 116.60454672 ... 106.92346514 103.11292712
   99.81820076]]
('*** Image : \n', array([[122, 118, 126, ..., 139, 133, 131],
       [125, 129, 136, ..., 138, 131, 131],
       [135, 137, 141, ..., 139, 133, 131],
       ...,
       [ 80,  85,  79, ...,  41,  38,  36],
       [ 80,  84,  80, ...,  34,  35,  32],
       [ 79,  75,  81, ...,  34,  38,  38]], dtype=uint8))
('*** Mean Sub Image: \n', array([[-17.95782017, -23.20099083, -14.43867488, ...,  11.22745162,
          7.32678952,   8.90422724],
       [-14.93802667, -11.56486712,  -4.9960876 , ...,   9.75243078,
          4.41955274,   6.90633392],
       [ -6.18779517,  -4.36709418,  -0.18916103, ...,  10.02974812,
          4.72962774,   4.97217335],
       ...,
       [-34.79470321, -31.43089638, -36.09183721, ..., -68.55090749,
        -69.38255857, -67.00710714],
       [-33.43096583, -31.97684971, -36.4367997 , ..., -74.97731271,
        -70.47284471, -69.36667747],
       [-32.82204371, -40.23474396, -35.60454672, ..., -72.92346514,
        -65.11292712, -61.81820076]]))
('*** Final cell Count:\n', [300, 300, 300, 0, 300, 300, 300, 300, 300, 300, 300, 300, 0, 0, 0, 300, 300, 300, 0, 0, 300, 296, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 0, 300, 0, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 0, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
*** Done. Saved train data as train_data-gray-re1.0-en275-max300-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-re1.0-en275-max300-submean.npy
*** Done. Saved train to one hot dict as train_onehotdict-gray-re1.0-en275-max300-submean.npy

071918 530p
/home/macalester/tensorflow/bin/python /home/macalester/PycharmProjects/olri_classifier/olin_inputs.py
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count before exclusion:\n', [574, 458, 494, 272, 430, 492, 368, 356, 324, 334, 366, 510, 256, 254, 264, 472, 544, 646, 0, 162, 606, 296, 1256, 732, 776, 694, 768, 370, 930, 1398, 1298, 1358, 1344, 1288, 1706, 1554, 1228, 1458, 1166, 1992, 1354, 1476, 1000, 302, 658, 576, 306, 992, 666, 566, 606, 548, 616, 626, 646, 688, 550, 574, 516, 574, 618, 532, 668, 548, 480, 444, 930, 528, 836, 764, 880, 756, 1336, 844, 794, 776, 894, 820, 882, 1522, 1312, 1212, 902, 946, 900, 1000, 1164, 1074, 776, 652, 606, 544, 714, 678, 596, 262, 984, 272, 1162, 520, 780, 522, 1132, 1112, 672, 684, 752, 684, 612, 852, 810, 876, 804, 1166, 508, 502, 572, 1024, 1070, 450, 676, 554, 824, 254, 694, 968, 1182, 762, 796, 802, 750, 760, 972, 532, 1060, 872, 812, 726, 774, 344, 1766, 872, 734, 1158, 614, 550, 594, 604, 558, 590, 610, 532, 460])
('*** Excluding cells that lack enough data (< 300): ', [3, 12, 13, 14, 18, 19, 21, 95, 97, 123])
('*** Cell Labels with len 143 contain these unique labels', set([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152]))
*** Limiting each cell to have maximum of 350 data
('*** Cell with too much data: ', [0, 1, 2, 4, 5, 6, 7, 10, 11, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152])
('*** Cell counts after limiting max :', [350, 350, 350, 0, 350, 350, 350, 350, 324, 334, 350, 350, 0, 0, 0, 350, 350, 350, 0, 0, 350, 0, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 302, 350, 350, 306, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 0, 350, 0, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 0, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 344, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350])
('*** int to one hot dict', {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 15: 11, 16: 12, 17: 13, 20: 14, 22: 15, 23: 16, 24: 17, 25: 18, 26: 19, 27: 20, 28: 21, 29: 22, 30: 23, 31: 24, 32: 25, 33: 26, 34: 27, 35: 28, 36: 29, 37: 30, 38: 31, 39: 32, 40: 33, 41: 34, 42: 35, 43: 36, 44: 37, 45: 38, 46: 39, 47: 40, 48: 41, 49: 42, 50: 43, 51: 44, 52: 45, 53: 46, 54: 47, 55: 48, 56: 49, 57: 50, 58: 51, 59: 52, 60: 53, 61: 54, 62: 55, 63: 56, 64: 57, 65: 58, 66: 59, 67: 60, 68: 61, 69: 62, 70: 63, 71: 64, 72: 65, 73: 66, 74: 67, 75: 68, 76: 69, 77: 70, 78: 71, 79: 72, 80: 73, 81: 74, 82: 75, 83: 76, 84: 77, 85: 78, 86: 79, 87: 80, 88: 81, 89: 82, 90: 83, 91: 84, 92: 85, 93: 86, 94: 87, 96: 88, 98: 89, 99: 90, 100: 91, 101: 92, 102: 93, 103: 94, 104: 95, 105: 96, 106: 97, 107: 98, 108: 99, 109: 100, 110: 101, 111: 102, 112: 103, 113: 104, 114: 105, 115: 106, 116: 107, 117: 108, 118: 109, 119: 110, 120: 111, 121: 112, 122: 113, 124: 114, 125: 115, 126: 116, 127: 117, 128: 118, 129: 119, 130: 120, 131: 121, 132: 122, 133: 123, 134: 124, 135: 125, 136: 126, 137: 127, 138: 128, 139: 129, 140: 130, 141: 131, 142: 132, 143: 133, 144: 134, 145: 135, 146: 136, 147: 137, 148: 138, 149: 139, 150: 140, 151: 141, 152: 142})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-re1.5-en300-max350-submean.npy
*** Mean subtraction normalization with the mean: 
[[139.87479463 141.03634542 140.26790222 ... 127.52021639 125.34762573
  121.69809657]
 [139.97679824 140.54728511 140.94640353 ... 127.83015428 126.21598878
  123.68320978]
 [141.21336406 141.37623723 141.04890803 ... 128.6347225  127.95976758
  125.72791024]
 ...
 [114.97743939 116.64075336 115.4408335  ... 109.4924464  107.38661591
  102.98916049]
 [113.59434983 116.33368063 116.70933681 ... 108.85353637 105.3931677
  101.2371669 ]
 [112.15670206 115.4895011  116.85780405 ... 106.86018834 102.99709477
   99.70024043]]
('*** Image : \n', array([[107, 112, 114, ...,   2,   8,  32],
       [109, 110, 112, ...,   2,  12,  31],
       [107, 108, 112, ...,   1,  11,  37],
       ...,
       [ 50,  48,   2, ...,  56,  54,  42],
       [ 48,   1,   3, ...,  51,  49,  40],
       [  2,   2,  16, ...,  51,  52,  46]], dtype=uint8))
('*** Mean Sub Image: \n', array([[ -32.87479463,  -29.03634542,  -26.26790222, ..., -125.52021639,
        -117.34762573,  -89.69809657],
       [ -30.97679824,  -30.54728511,  -28.94640353, ..., -125.83015428,
        -114.21598878,  -92.68320978],
       [ -34.21336406,  -33.37623723,  -29.04890803, ..., -127.6347225 ,
        -116.95976758,  -88.72791024],
       ...,
       [ -64.97743939,  -68.64075336, -113.4408335 , ...,  -53.4924464 ,
         -53.38661591,  -60.98916049],
       [ -65.59434983, -115.33368063, -113.70933681, ...,  -57.85353637,
         -56.3931677 ,  -61.2371669 ],
       [-110.15670206, -113.4895011 , -100.85780405, ...,  -55.86018834,
         -50.99709477,  -53.70024043]]))
('*** Final cell Count:\n', [350, 350, 350, 0, 350, 350, 350, 350, 324, 334, 350, 350, 0, 0, 0, 350, 350, 350, 0, 0, 350, 0, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 302, 350, 350, 306, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 0, 350, 0, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 0, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 344, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350])
*** Done. Saved train data as train_data-gray-re1.5-en300-max350-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-re1.5-en300-max350-submean.npy
*** Done. Saved train to one hot dict as train_onehotdict-gray-re1.5-en300-max350-submean.npy


071618 600p,
/home/macalester/tensorflow/bin/python /home/macalester/PycharmProjects/olri_classifier/olin_inputs.py
('Cell Num', 135)
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count before exclusion:\n', [574, 458, 494, 272, 430, 492, 368, 356, 324, 334, 366, 510, 256, 254, 264, 472, 544, 646, 0, 162, 606, 296, 1256, 732, 776, 694, 768, 370, 930, 1398, 1298, 1358, 1344, 1288, 1706, 1554, 1228, 1458, 1166, 1992, 1354, 1476, 1000, 302, 658, 576, 306, 992, 666, 566, 606, 548, 616, 626, 646, 688, 550, 574, 516, 574, 618, 532, 668, 548, 480, 444, 274, 76, 314, 232, 304, 226, 774, 246, 278, 234, 262, 284, 272, 744, 400, 556, 252, 258, 284, 344, 498, 736, 370, 356, 306, 294, 364, 344, 336, 32, 734, 36, 682, 178, 386, 92, 478, 280, 114, 120, 138, 148, 90, 0, 0, 0, 0, 96, 140, 134, 122, 176, 318, 194, 426, 168, 282, 190, 260, 130, 244, 190, 178, 378, 282, 302, 506, 222, 660, 404, 358, 234, 312, 252, 718, 376, 236, 660, 128, 156, 152, 170, 144, 132, 160, 124, 460])
('*** Excluding cells that lack enough data (< 250): ', [18, 19, 67, 69, 71, 73, 75, 95, 97, 99, 101, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 121, 123, 125, 126, 127, 128, 133, 137, 142, 144, 145, 146, 147, 148, 149, 150, 151])
('*** Cell Labels with len 110 contain these unique labels', set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 70, 72, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 100, 102, 103, 118, 120, 122, 124, 129, 130, 131, 132, 134, 135, 136, 138, 139, 140, 141, 143, 152]))
*** Limiting each cell to have maximum of 300 data
('*** Cell with too much data: ', [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 68, 70, 72, 79, 80, 81, 85, 86, 87, 88, 89, 90, 92, 93, 94, 96, 98, 100, 102, 118, 120, 129, 131, 132, 134, 135, 136, 138, 140, 141, 143, 152])
('*** Cell counts after limiting max :', [300, 300, 300, 272, 300, 300, 300, 300, 300, 300, 300, 300, 256, 254, 264, 300, 300, 300, 0, 0, 300, 296, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 274, 0, 300, 0, 300, 0, 300, 0, 278, 0, 262, 284, 272, 300, 300, 300, 252, 258, 284, 300, 300, 300, 300, 300, 300, 294, 300, 300, 300, 0, 300, 0, 300, 0, 300, 0, 300, 280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 300, 0, 300, 0, 282, 0, 260, 0, 0, 0, 0, 300, 282, 300, 300, 0, 300, 300, 300, 0, 300, 252, 300, 300, 0, 300, 0, 0, 0, 0, 0, 0, 0, 0, 300])
('*** int to one hot dict', {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 34: 32, 35: 33, 36: 34, 37: 35, 38: 36, 39: 37, 40: 38, 41: 39, 42: 40, 43: 41, 44: 42, 45: 43, 46: 44, 47: 45, 48: 46, 49: 47, 50: 48, 51: 49, 52: 50, 53: 51, 54: 52, 55: 53, 56: 54, 57: 55, 58: 56, 59: 57, 60: 58, 61: 59, 62: 60, 63: 61, 64: 62, 65: 63, 66: 64, 68: 65, 70: 66, 72: 67, 74: 68, 76: 69, 77: 70, 78: 71, 79: 72, 80: 73, 81: 74, 82: 75, 83: 76, 84: 77, 85: 78, 86: 79, 87: 80, 88: 81, 89: 82, 90: 83, 91: 84, 92: 85, 93: 86, 94: 87, 96: 88, 98: 89, 100: 90, 102: 91, 103: 92, 118: 93, 120: 94, 122: 95, 124: 96, 129: 97, 130: 98, 131: 99, 132: 100, 134: 101, 135: 102, 136: 103, 138: 104, 139: 105, 140: 106, 141: 107, 143: 108, 152: 109})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-re1.0-en250-max300-submean.npy
*** Mean subtraction normalization with the mean: 
[[141.28777422 142.01827089 141.09252526 ... 129.01568277 126.55170076
  122.60176855]
 [141.42324994 141.72245502 141.80542889 ... 129.6548558  127.63874168
  124.86156643]
 [142.493807   142.74439241 142.36024156 ... 130.38211733 129.524772
  127.12761893]
 ...
 [115.46866527 117.27970175 115.83371333 ... 109.02455632 106.69383165
  102.40975474]
 [114.18693    117.01965738 117.33112522 ... 108.27886985 104.87238107
  100.81599704]
 [112.88507518 116.23228371 117.32043382 ... 106.33204954 102.48345452
   99.09927286]]
('*** Image : \n', array([[119, 137,  47, ...,  15,  19,  18],
       [137, 113, 130, ...,  17,  20,  21],
       [139, 136, 134, ...,  19,  22,  22],
       ...,
       [123, 126, 123, ..., 144, 140, 130],
       [128, 127, 134, ..., 138, 139, 131],
       [119, 127, 131, ..., 138, 127, 129]], dtype=uint8))
('*** Mean Sub Image: \n', array([[ -22.28777422,   -5.01827089,  -94.09252526, ..., -114.01568277,
        -107.55170076, -104.60176855],
       [  -4.42324994,  -28.72245502,  -11.80542889, ..., -112.6548558 ,
        -107.63874168, -103.86156643],
       [  -3.493807  ,   -6.74439241,   -8.36024156, ..., -111.38211733,
        -107.524772  , -105.12761893],
       ...,
       [   7.53133473,    8.72029825,    7.16628667, ...,   34.97544368,
          33.30616835,   27.59024526],
       [  13.81307   ,    9.98034262,   16.66887478, ...,   29.72113015,
          34.12761893,   30.18400296],
       [   6.11492482,   10.76771629,   13.67956618, ...,   31.66795046,
          24.51654548,   29.90072714]]))
('*** Final cell Count:\n', [300, 300, 300, 272, 300, 300, 300, 300, 300, 300, 300, 300, 256, 254, 264, 300, 300, 300, 0, 0, 300, 296, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 274, 0, 300, 0, 300, 0, 300, 0, 278, 0, 262, 284, 272, 300, 300, 300, 252, 258, 284, 300, 300, 300, 300, 300, 300, 294, 300, 300, 300, 0, 300, 0, 300, 0, 300, 0, 300, 280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 300, 0, 300, 0, 282, 0, 260, 0, 0, 0, 0, 300, 282, 300, 300, 0, 300, 300, 300, 0, 300, 252, 300, 300, 0, 300, 0, 0, 0, 0, 0, 0, 0, 0, 300])
*** Done. Saved train data as train_data-gray-re1.0-en250-max300-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-re1.0-en250-max300-submean.npy
*** Done. Saved train to one hot dict as train_onehotdict-gray-re1.0-en250-max300-submean.npy

071318 330pm
/home/macalester/tensorflow/bin/python /home/macalester/PycharmProjects/olri_classifier/olin_inputs.py
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count before exclusion:\n', [186, 124, 96, 98, 112, 86, 88, 106, 76, 88, 114, 282, 256, 254, 226, 234, 260, 112, 0, 8, 252, 116, 860, 732, 776, 694, 768, 370, 930, 1398, 1298, 1358, 1344, 1288, 1706, 1554, 1228, 1458, 1166, 1992, 1354, 1476, 1000, 302, 658, 576, 306, 992, 666, 566, 606, 548, 616, 626, 646, 688, 550, 574, 516, 574, 618, 532, 668, 548, 480, 444, 274, 76, 314, 232, 304, 226, 774, 246, 278, 234, 262, 284, 272, 478, 144, 556, 252, 258, 284, 344, 498, 736, 370, 356, 306, 294, 364, 344, 336, 32, 734, 36, 682, 178, 386, 92, 478, 280, 114, 120, 138, 148, 90, 0, 0, 0, 0, 96, 140, 134, 122, 176, 318, 194, 426, 168, 282, 190, 260, 130, 244, 190, 178, 378, 282, 302, 506, 222, 660, 404, 358, 234, 312, 252, 718, 376, 236, 660, 128, 156, 152, 170, 144, 132, 160, 124, 460])
('*** Excluding cells that lack enough data (< 100): ', [2, 3, 5, 6, 8, 9, 18, 19, 67, 95, 97, 101, 108, 109, 110, 111, 112, 113])
('*** Cell Labels with len 135 contain these unique labels', set([0, 1, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 102, 103, 104, 105, 106, 107, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152]))
*** Limiting each cell to have maximum of 150 data
('*** Cell with too much data: ', [0, 11, 12, 13, 14, 15, 16, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 102, 103, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 150, 152])
('*** Cell counts after limiting max :', [150, 124, 0, 0, 112, 0, 0, 106, 0, 0, 114, 150, 150, 150, 150, 150, 150, 112, 0, 0, 150, 116, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 0, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 144, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 0, 150, 0, 150, 150, 150, 0, 150, 150, 114, 120, 138, 148, 0, 0, 0, 0, 0, 0, 140, 134, 122, 150, 150, 150, 150, 150, 150, 150, 150, 130, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 128, 150, 150, 150, 144, 132, 150, 124, 150])
('*** int to one hot dict', {0: 0, 1: 1, 4: 2, 7: 3, 10: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 25: 17, 26: 18, 27: 19, 28: 20, 29: 21, 30: 22, 31: 23, 32: 24, 33: 25, 34: 26, 35: 27, 36: 28, 37: 29, 38: 30, 39: 31, 40: 32, 41: 33, 42: 34, 43: 35, 44: 36, 45: 37, 46: 38, 47: 39, 48: 40, 49: 41, 50: 42, 51: 43, 52: 44, 53: 45, 54: 46, 55: 47, 56: 48, 57: 49, 58: 50, 59: 51, 60: 52, 61: 53, 62: 54, 63: 55, 64: 56, 65: 57, 66: 58, 68: 59, 69: 60, 70: 61, 71: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 83: 74, 84: 75, 85: 76, 86: 77, 87: 78, 88: 79, 89: 80, 90: 81, 91: 82, 92: 83, 93: 84, 94: 85, 96: 86, 98: 87, 99: 88, 100: 89, 102: 90, 103: 91, 104: 92, 105: 93, 106: 94, 107: 95, 114: 96, 115: 97, 116: 98, 117: 99, 118: 100, 119: 101, 120: 102, 121: 103, 122: 104, 123: 105, 124: 106, 125: 107, 126: 108, 127: 109, 128: 110, 129: 111, 130: 112, 131: 113, 132: 114, 133: 115, 134: 116, 135: 117, 136: 118, 137: 119, 138: 120, 139: 121, 140: 122, 141: 123, 142: 124, 143: 125, 144: 126, 145: 127, 146: 128, 147: 129, 148: 130, 149: 131, 150: 132, 151: 133, 152: 134})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-re1.0-en100-max150-submean.npy
*** Mean subtraction normalization with the mean: 
[[137.57110393 138.5028785  137.86496314 ... 126.36056964 123.9981315
  120.23916776]
 [138.12266438 138.23972326 138.5985254  ... 126.55287345 124.69119281
  122.08579941]
 [139.14705585 139.17154833 138.77219473 ... 127.65947884 126.83840016
  124.32027068]
 ...
 [119.05938794 120.68063832 119.11917988 ... 110.77845672 108.76224624
  104.22750227]
 [117.80648419 120.09261691 120.23613776 ... 110.23775376 106.5994849
  102.43732956]
 [116.15175235 119.22528027 120.11205939 ... 108.15043935 104.06968993
  100.67654782]]
('*** Image : \n', array([[224, 222, 220, ..., 198, 202, 205],
       [225, 225, 222, ..., 191, 186, 184],
       [229, 226, 224, ..., 183, 178, 175],
       ...,
       [126, 115, 127, ..., 162, 169, 169],
       [104, 132, 118, ..., 162, 173, 171],
       [104, 110, 117, ..., 132, 132, 129]], dtype=uint8))
('*** Mean Sub Image: \n', array([[ 86.42889607,  83.4971215 ,  82.13503686, ...,  71.63943036,
         78.0018685 ,  84.76083224],
       [ 86.87733562,  86.76027674,  83.4014746 , ...,  64.44712655,
         61.30880719,  61.91420059],
       [ 89.85294415,  86.82845167,  85.22780527, ...,  55.34052116,
         51.16159984,  50.67972932],
       ...,
       [  6.94061206,  -5.68063832,   7.88082012, ...,  51.22154328,
         60.23775376,  64.77249773],
       [-13.80648419,  11.90738309,  -2.23613776, ...,  51.76224624,
         66.4005151 ,  68.56267044],
       [-12.15175235,  -9.22528027,  -3.11205939, ...,  23.84956065,
         27.93031007,  28.32345218]]))
('*** Final cell Count:\n', [150, 124, 0, 0, 112, 0, 0, 106, 0, 0, 114, 150, 150, 150, 150, 150, 150, 112, 0, 0, 150, 116, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 0, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 144, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 0, 150, 0, 150, 150, 150, 0, 150, 150, 114, 120, 138, 148, 0, 0, 0, 0, 0, 0, 140, 134, 122, 150, 150, 150, 150, 150, 150, 150, 150, 130, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 128, 150, 150, 150, 144, 132, 150, 124, 150])
*** Done. Saved train data as train_data-gray-re1.0-en100-max150-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-re1.0-en100-max150-submean.npy
*** Done. Saved train to one hot dict as train_onehotdict-gray-re1.0-en100-max150-submean.npy


071318 1pm
/home/macalester/tensorflow/bin/python /home/macalester/PycharmProjects/olri_classifier/olin_inputs.py
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count:\n', [186, 124, 96, 98, 112, 86, 88, 106, 76, 88, 114, 282, 256, 254, 226, 
234, 260, 112, 0, 8, 252, %116, %860, %732, %776, %694, %768, 370, 930, 1398, 1298, 1358, 1344, 
1288, 1706, 1554, 1228, 1458, 1166, 1992, 1354, 1476, 1000, 302, %658, %576, %306, %992, %666, 
%566, %606, %548, %616, %626, %646, %688, %550, %574, %516, %574, %618, %532, %668, %548, %480, %444, 274, 
76, 314, 232, 304, 226, 774, 246, 278, 234, 262, 284, 272, 478, 144, 556, 252, 258, 284, 
344, 498, 736, 370, 356, 306, 294, 364, 344, 336, 32, 734, 36, 682, 178, 386, 92, 478, 280, 
114, 120, 138, 148, 90, 0, 0, 0, 0, 96, 140, 134, 122, 176, 318, 194, 426, 168, 282, 190, 
260, 130, 244, 190, 178, 378, 282, 302, 506, 222, 660, 404, 358, 234, 312, 252, 718, 376, 
236, 660, 128, 156, 152, 170, 144, 132, 160, 124, 460])
('*** Excluding cells that lack enough data (< 100): ', [2, 3, 5, 6, 8, 9, 18, 19, 67, 95, 97, 101, 108, 109, 110, 111, 112, 113])
('*** Cell Labels with len 64240 contain these unique labels', set([0, 1, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 102, 103, 104, 105, 106, 107, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152]))
('*** int to one hot dict', {0: 0, 1: 1, 4: 2, 7: 3, 10: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 25: 17, 26: 18, 27: 19, 28: 20, 29: 21, 30: 22, 31: 23, 32: 24, 33: 25, 34: 26, 35: 27, 36: 28, 37: 29, 38: 30, 39: 31, 40: 32, 41: 33, 42: 34, 43: 35, 44: 36, 45: 37, 46: 38, 47: 39, 48: 40, 49: 41, 50: 42, 51: 43, 52: 44, 53: 45, 54: 46, 55: 47, 56: 48, 57: 49, 58: 50, 59: 51, 60: 52, 61: 53, 62: 54, 63: 55, 64: 56, 65: 57, 66: 58, 68: 59, 69: 60, 70: 61, 71: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 83: 74, 84: 75, 85: 76, 86: 77, 87: 78, 88: 79, 89: 80, 90: 81, 91: 82, 92: 83, 93: 84, 94: 85, 96: 86, 98: 87, 99: 88, 100: 89, 102: 90, 103: 91, 104: 92, 105: 93, 106: 94, 107: 95, 114: 96, 115: 97, 116: 98, 117: 99, 118: 100, 119: 101, 120: 102, 121: 103, 122: 104, 123: 105, 124: 106, 125: 107, 126: 108, 127: 109, 128: 110, 129: 111, 130: 112, 131: 113, 132: 114, 133: 115, 134: 116, 135: 117, 136: 118, 137: 119, 138: 120, 139: 121, 140: 122, 141: 123, 142: 124, 143: 125, 144: 126, 145: 127, 146: 128, 147: 129, 148: 130, 149: 131, 150: 132, 151: 133, 152: 134})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-re1.0-en100-submean.npy
*** Mean subtraction normalization with the mean: 
[[140.08765567 141.34190535 141.09788294 ... 129.79764944 127.20446762
  123.08812267]
 [140.59220112 141.1235056  141.52781756 ... 130.4378736  128.16354296
  125.28111768]
 [141.76021171 141.99486301 141.96471046 ... 131.285632   130.12786426
  127.37255604]
 ...
 [123.64741594 125.59419365 124.11365193 ... 116.24193649 113.90969801
  109.25409402]
 [122.37800436 125.15941781 125.44123599 ... 115.58919676 111.9374066
  107.50733188]
 [120.75731631 124.2882005  125.55235056 ... 113.6255604  109.41844645
  105.65379826]]
('*** Image : \n', array([[123, 129, 134, ...,  21,  24,  23],
       [123, 130, 131, ...,  24,  25,  19],
       [115, 122, 125, ...,  27,  22,  18],
       ...,
       [126, 132, 134, ...,  95,  99,  87],
       [131, 141, 120, ..., 102, 103,  94],
       [121, 109,  89, ..., 103,  99,  91]], dtype=uint8))
('*** Mean Sub Image: \n', array([[ -17.08765567,  -12.34190535,   -7.09788294, ..., -108.79764944,
        -103.20446762, -100.08812267],
       [ -17.59220112,  -11.1235056 ,  -10.52781756, ..., -106.4378736 ,
        -103.16354296, -106.28111768],
       [ -26.76021171,  -19.99486301,  -16.96471046, ..., -104.285632  ,
        -108.12786426, -109.37255604],
       ...,
       [   2.35258406,    6.40580635,    9.88634807, ...,  -21.24193649,
         -14.90969801,  -22.25409402],
       [   8.62199564,   15.84058219,   -5.44123599, ...,  -13.58919676,
          -8.9374066 ,  -13.50733188],
       [   0.24268369,  -15.2882005 ,  -36.55235056, ...,  -10.6255604 ,
         -10.41844645,  -14.65379826]]))
('*** Cell Count:\n', [186, 124, 0, 0, 112, 0, 0, 106, 0, 0, 114, 282, 256, 254, 226, 234, 260, 112, 0, 0, 252, 116, 860, 732, 776, 694, 768, 370, 930, 1398, 1298, 1358, 1344, 1288, 1706, 1554, 1228, 1458, 1166, 1992, 1354, 1476, 1000, 302, 658, 576, 306, 992, 666, 566, 606, 548, 616, 626, 646, 688, 550, 574, 516, 574, 618, 532, 668, 548, 480, 444, 274, 0, 314, 232, 304, 226, 774, 246, 278, 234, 262, 284, 272, 478, 144, 556, 252, 258, 284, 344, 498, 736, 370, 356, 306, 294, 364, 344, 336, 0, 734, 0, 682, 178, 386, 0, 478, 280, 114, 120, 138, 148, 0, 0, 0, 0, 0, 0, 140, 134, 122, 176, 318, 194, 426, 168, 282, 190, 260, 130, 244, 190, 178, 378, 282, 302, 506, 222, 660, 404, 358, 234, 312, 252, 718, 376, 236, 660, 128, 156, 152, 170, 144, 132, 160, 124, 460])
*** Done. Saved train data as train_data-gray-re1.0-en100-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-re1.0-en100-submean.npy




071318 1pm
/home/macalester/tensorflow/bin/python /home/macalester/PycharmProjects/olri_classifier/olin_inputs.py
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count:\n', [186, 124, 96, 98, 112, 86, 88, 106, 76, 88, 114, 282, 256, 254, 226, 
234, 260, 112, 0, 8, 252, 116, 488, 336, 326, 274, 344, 370, 930, 1398, 1298, 1358, 1344, 
1288, 1706, 1554, 978, 1458, 1166, 1992, 1354, 1476, 1000, 302, 222, 216, 224, 264, 288, 
214, 234, 220, 228, 208, 224, 216, 210, 170, 140, 182, 166, 162, 254, 214, 192, 226, 274, 
76, 314, 232, 304, 226, 774, 246, 278, 234, 262, 284, 272, 478, 144, 556, 252, 258, 284, 
344, 498, 736, 370, 356, 306, 294, 364, 344, 336, 32, 734, 36, 682, 178, 386, 92, 478, 
280, 114, 120, 138, 148, 90, 0, 0, 0, 0, 96, 140, 134, 122, 176, 318, 194, 426, 168, 282, 
190, 260, 130, 244, 190, 178, 378, 282, 302, 506, 222, 660, 404, 358, 234, 312, 252, 718, 
376, 236, 660, 128, 156, 152, 170, 144, 132, 160, 124, 460])
('*** Excluding cells that lack enough data (< 597.6): ', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152])
('*** Cell Labels with len 25264 contain these unique labels', set([134, 140, 143, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 72, 87, 96, 98]))
('*** int to one hot dict', {134: 0, 140: 1, 143: 2, 28: 3, 29: 4, 30: 5, 31: 6, 32: 7, 33: 8, 34: 9, 35: 10, 36: 11, 37: 12, 38: 13, 39: 14, 40: 15, 41: 16, 42: 17, 72: 18, 87: 19, 96: 20, 98: 21})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-re0.7-er0.3-submean.npy
*** Mean subtraction normalization with the mean: 
[[138.91070298 140.73709626 140.58058898 ... 127.38192685 124.64443477
  120.68211685]
 [139.5270741  140.74980209 141.20859721 ... 128.26167669 126.4291482
  123.26413078]
 [141.13453927 141.88707251 141.75292907 ... 129.26785149 128.01203293
  124.97621121]
 ...
 [126.98638379 129.16359246 127.85077581 ... 119.57140595 117.10065706
  112.2598955 ]
 [125.50221659 128.65191577 129.03297182 ... 118.50938094 114.97114471
  110.3415532 ]
 [123.77248258 127.66066339 129.24612096 ... 116.7543936  112.47961526
  108.51314123]]
('*** Image : \n', array([[189, 192, 194, ..., 141, 141, 137],
       [191, 193, 192, ..., 142, 138, 137],
       [193, 191, 193, ..., 146, 141, 132],
       ...,
       [129, 130, 133, ..., 135, 133, 128],
       [124, 125, 136, ..., 136, 130, 125],
       [130, 139, 134, ..., 134, 126, 119]], dtype=uint8))
('*** Mean Sub Image: \n', array([[50.08929702, 51.26290374, 53.41941102, ..., 13.61807315,
        16.35556523, 16.31788315],
       [51.4729259 , 52.25019791, 50.79140279, ..., 13.73832331,
        11.5708518 , 13.73586922],
       [51.86546073, 49.11292749, 51.24707093, ..., 16.73214851,
        12.98796707,  7.02378879],
       ...,
       [ 2.01361621,  0.83640754,  5.14922419, ..., 15.42859405,
        15.89934294, 15.7401045 ],
       [-1.50221659, -3.65191577,  6.96702818, ..., 17.49061906,
        15.02885529, 14.6584468 ],
       [ 6.22751742, 11.33933661,  4.75387904, ..., 17.2456064 ,
        13.52038474, 10.48685877]]))
('*** Cell Count:\n', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 930, 1398, 1298, 1358, 1344, 1288, 1706, 1554, 978, 1458, 1166, 1992, 1354, 1476, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 774, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 736, 0, 0, 0, 0, 0, 0, 0, 0, 734, 0, 682, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 660, 0, 0, 0, 0, 0, 718, 0, 0, 660, 0, 0, 0, 0, 0, 0, 0, 0, 0])
*** Done. Saved train data as train_data-gray-re0.7-er0.3-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-re0.7-er0.3-submean.npy


071318 12pm
/home/macalester/tensorflow/bin/python /home/macalester/PycharmProjects/olri_classifier/olin_inputs.py
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count:\n', [93, 62, 48, 49, 56, 43, 44, 53, 38, 44, 57, 141, 128, 127, 113, 117, 130, 56, 0, 4, 126, 58, 244, 168, 163, 137, 172, 185, 465, 699, 649, 679, 672, 644, 853, 777, 489, 729, 583, 996, 677, 738, 500, 151, 111, 108, 112, 132, 144, 107, 117, 110, 114, 104, 112, 108, 105, 85, 70, 91, 83, 81, 127, 107, 96, 113, 137, 38, 157, 116, 152, 113, 387, 123, 139, 117, 131, 142, 136, 239, 72, 278, 126, 129, 142, 172, 249, 368, 185, 178, 153, 147, 182, 172, 168, 16, 367, 18, 341, 89, 193, 46, 239, 140, 57, 60, 69, 74, 45, 0, 0, 0, 0, 48, 70, 67, 61, 88, 159, 97, 213, 84, 141, 95, 130, 65, 122, 95, 89, 189, 141, 151, 253, 111, 330, 202, 179, 117, 156, 126, 359, 188, 118, 330, 64, 78, 76, 85, 72, 66, 80, 62, 230])
('*** Excluding cells that lack enough data (< 298.8): ', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152])
('*** Cell Labels with len 12632 contain these unique labels', set([134, 140, 143, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 72, 87, 96, 98]))
('*** int to one hot dict', {134: 0, 140: 1, 143: 2, 28: 3, 29: 4, 30: 5, 31: 6, 32: 7, 33: 8, 34: 9, 35: 10, 36: 11, 37: 12, 38: 13, 39: 14, 40: 15, 41: 16, 42: 17, 72: 18, 87: 19, 96: 20, 98: 21})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-er0.3-submean.npy
*** Mean subtraction normalization with the mean: 
[[138.91070298 140.74390437 140.58114313 ... 127.39241609 124.64946168
  120.68429386]
 [139.54235275 140.76717859 141.22450918 ... 128.26971184 126.42241925
  123.25918303]
 [141.14407853 141.90381571 141.77462001 ... 129.27082014 128.00047498
  124.96508866]
 ...
 [127.00071248 129.17764408 127.86185877 ... 119.53871121 117.06705193
  112.24430019]
 [125.50625396 128.65745725 129.03752375 ... 118.49572514 114.95693477
  110.3298765 ]
 [123.77636162 127.66236542 129.25079164 ... 116.74984167 112.47181761
  108.50292907]]
('*** Image : \n', array([[189, 192, 194, ..., 141, 141, 137],
       [191, 193, 192, ..., 142, 138, 137],
       [193, 191, 193, ..., 146, 141, 132],
       ...,
       [129, 130, 133, ..., 135, 133, 128],
       [124, 125, 136, ..., 136, 130, 125],
       [130, 139, 134, ..., 134, 126, 119]], dtype=uint8))
('*** Mean Sub Image: \n', array([[50.08929702, 51.25609563, 53.41885687, ..., 13.60758391,
        16.35053832, 16.31570614],
       [51.45764725, 52.23282141, 50.77549082, ..., 13.73028816,
        11.57758075, 13.74081697],
       [51.85592147, 49.09618429, 51.22537999, ..., 16.72917986,
        12.99952502,  7.03491134],
       ...,
       [ 1.99928752,  0.82235592,  5.13814123, ..., 15.46128879,
        15.93294807, 15.75569981],
       [-1.50625396, -3.65745725,  6.96247625, ..., 17.50427486,
        15.04306523, 14.6701235 ],
       [ 6.22363838, 11.33763458,  4.74920836, ..., 17.25015833,
        13.52818239, 10.49707093]]))
('*** Cell Count:\n', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 465, 699, 649, 679, 672, 644, 853, 777, 489, 729, 583, 996, 677, 738, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 387, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 368, 0, 0, 0, 0, 0, 0, 0, 0, 367, 0, 341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 330, 0, 0, 0, 0, 0, 359, 0, 0, 330, 0, 0, 0, 0, 0, 0, 0, 0, 0])
*** Done. Saved train data as train_data-gray-er0.3-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-er0.3-submean.npy



071218 2pm
*** Processing train data... this may take a few minutes...
*** Reading cell data and returning as a dictionary...
('*** Cell Count:\n', [93, 62, 48, 49, 56, 43, 44, 53, 38, 44, 57, 141, 128, 127, 113, 117, 130, 56, 0, 4, 126, 58, 244, 168, 163, 137, 172, 185, 465, 699, 649, 679, 672, 644, 853, 777, 489, 729, 583, 996, 677, 738, 500, 151, 111, 108, 112, 132, 144, 107, 117, 110, 114, 104, 112, 108, 105, 85, 70, 91, 83, 81, 127, 107, 96, 113, 137, 38, 157, 116, 152, 113, 387, 123, 139, 117, 131, 142, 136, 239, 72, 278, 126, 129, 142, 172, 249, 368, 185, 178, 153, 147, 182, 172, 168, 16, 367, 18, 341, 89, 193, 46, 239, 140, 57, 60, 69, 74, 45, 0, 0, 0, 0, 48, 70, 67, 61, 88, 159, 97, 213, 84, 141, 95, 130, 65, 122, 95, 89, 189, 141, 151, 253, 111, 330, 202, 179, 117, 156, 126, 359, 188, 118, 330, 64, 78, 76, 85, 72, 66, 80, 62, 230])
('*** Excluding cells that lack enough data (< 199.2): ', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 82, 83, 84, 85, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 136, 137, 138, 139, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151])
('*** Cell Labels with len 14779 contain these unique labels', set([132, 134, 135, 140, 143, 22, 152, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 72, 79, 81, 86, 87, 96, 98, 102, 120]))
('*** int to one hot dict', {132: 0, 134: 1, 135: 2, 140: 3, 143: 4, 22: 5, 152: 6, 28: 7, 29: 8, 30: 9, 31: 10, 32: 11, 33: 12, 34: 13, 35: 14, 36: 15, 37: 16, 38: 17, 39: 18, 40: 19, 41: 20, 42: 21, 72: 22, 79: 23, 81: 24, 86: 25, 87: 26, 96: 27, 98: 28, 102: 29, 120: 30})
*** Done. Returning mean.
*** Saved mean as...train_mean-gray-er0.2-submean.npy
*** Mean subtraction normalization with the mean: 
[[139.26760945 140.21009541 139.78740104 ... 128.27802964 125.85438798
  121.23289803]
 [141.02381758 141.17937614 140.67514717 ... 129.64733744 127.124095
  123.80411395]
 [141.07267068 141.40983828 139.77901076 ... 128.66492997 126.919751
  124.28168347]
 ...
 [126.06976115 126.0371473  125.6234522  ... 119.54178226 117.01292374
  112.55923946]
 [124.52696394 127.20265241 125.16212193 ... 118.51573178 115.3249205
  110.27992422]
 [123.02462954 126.77217674 126.15711483 ... 116.86812369 112.63333108
  108.0650247 ]]
('*** Image : \n', array([[192, 196, 194, ..., 149, 144, 138],
       [192, 193, 193, ..., 144, 139, 135],
       [193, 192, 192, ..., 137, 139, 134],
       ...,
       [131, 133, 130, ..., 137, 142, 133],
       [132, 128, 137, ..., 143, 128, 130],
       [131, 141, 132, ..., 135, 129, 121]], dtype=uint8))
('*** Mean Sub Image: \n', array([[52.73239055, 55.78990459, 54.21259896, ..., 20.72197036,
        18.14561202, 16.76710197],
       [50.97618242, 51.82062386, 52.32485283, ..., 14.35266256,
        11.875905  , 11.19588605],
       [51.92732932, 50.59016172, 52.22098924, ...,  8.33507003,
        12.080249  ,  9.71831653],
       ...,
       [ 4.93023885,  6.9628527 ,  4.3765478 , ..., 17.45821774,
        24.98707626, 20.44076054],
       [ 7.47303606,  0.79734759, 11.83787807, ..., 24.48426822,
        12.6750795 , 19.72007578],
       [ 7.97537046, 14.22782326,  5.84288517, ..., 18.13187631,
        16.36666892, 12.9349753 ]]))
('*** Cell Count:\n', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 244, 0, 0, 0, 0, 0, 465, 699, 649, 679, 672, 644, 853, 777, 489, 729, 583, 996, 677, 738, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 387, 0, 0, 0, 0, 0, 0, 239, 0, 278, 0, 0, 0, 0, 249, 368, 0, 0, 0, 0, 0, 0, 0, 0, 367, 0, 341, 0, 0, 0, 239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 0, 330, 202, 0, 0, 0, 0, 359, 0, 0, 330, 0, 0, 0, 0, 0, 0, 0, 0, 230])
*** Done. Saved train data as train_data-gray-er0.2-submean.npy
*** Done. Saved train class count array as train_cellcounts-gray-er0.2-submean.npy


071018 2pm
Processing train data... this may take a few minutes...
Reading cell data and returning as a dictionary...
('*** Excluding cells that lack enough data: ', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 36, 38, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152])
('*** Cell Labels with len 4224 contain these unique labels', set([32, 33, 34, 35, 37, 39, 40, 41, 29, 30, 31]))
('*** int to one hot dict', {32: 0, 33: 1, 34: 2, 35: 3, 37: 4, 39: 5, 40: 6, 41: 7, 29: 8, 30: 9, 31: 10})
*** Done. Returning mean.
*** Saved mean as...LESS071018train_mean-gray-er0.6-submean.npy
*** Mean subtraction normalization with the mean: 
[[141.85724432 143.91169508 143.38139205 ... 129.0530303  126.67447917
  122.83309659]
 [143.50213068 145.11316288 144.81960227 ... 131.08143939 129.43868371
  126.30042614]
 [143.50331439 144.78622159 143.23910985 ... 130.74431818 129.42732008
  126.42637311]
 ...
 [126.16856061 126.46756629 125.69412879 ... 120.46117424 117.84232955
  112.87689394]
 [124.79403409 127.96543561 125.58120265 ... 119.5390625  115.97466856
  110.55965909]
 [123.47064394 127.4827178  126.66122159 ... 117.97632576 113.30752841
  108.23508523]]
('*** Image : \n', array([[115, 118, 117, ..., 138, 136, 130],
       [115, 117, 113, ..., 137, 134, 134],
       [125, 119, 117, ..., 137, 138, 135],
       ...,
       [185, 177, 181, ..., 100,  72,  51],
       [183, 178, 180, ...,  84,  93,  88],
       [176, 189, 183, ..., 101,  97,  75]], dtype=uint8))
('*** Mean Sub Image: \n', array([[-26.85724432, -25.91169508, -26.38139205, ...,   8.9469697 ,
          9.32552083,   7.16690341],
       [-28.50213068, -28.11316288, -31.81960227, ...,   5.91856061,
          4.56131629,   7.69957386],
       [-18.50331439, -25.78622159, -26.23910985, ...,   6.25568182,
          8.57267992,   8.57362689],
       ...,
       [ 58.83143939,  50.53243371,  55.30587121, ..., -20.46117424,
        -45.84232955, -61.87689394],
       [ 58.20596591,  50.03456439,  54.41879735, ..., -35.5390625 ,
        -22.97466856, -22.55965909],
       [ 52.52935606,  61.5172822 ,  56.33877841, ..., -16.97632576,
        -16.30752841, -33.23508523]]))
('*** Cell Count:\n', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 363, 335, 340, 325, 331, 506, 345, 0, 386, 0, 456, 394, 443, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
*** Done. Saved train data as LESS071018train_data-gray-er0.6-submean.npy
*** Done. Saved train class count array as train_classweight_gray-intlabel-submean.npy




071018 12am
Processing train data... this may take a few minutes...
Reading cell data and returning as a dictionary...
('*** Excluding cells that lack enough data: ', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 36, 38, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152])
*** Resetting cell_counts AND deleting images and cell_labels that are 0th, modified to (cell number of 0)
*** Resetting cell_counts AND deleting images and cell_labels that are 3th, modified to (cell number of 2)
*** Resetting cell_counts AND deleting images and cell_labels that are 4th, modified to (cell number of 2)
*** Resetting cell_counts AND deleting images and cell_labels that are 7th, modified to (cell number of 4)
*** Resetting cell_counts AND deleting images and cell_labels that are 8th, modified to (cell number of 4)
*** Resetting cell_counts AND deleting images and cell_labels that are 10th, modified to (cell number of 5)
...
*** Done. Returning mean.
*** Saved mean as...LESS071018train_mean-gray-er0.6-submean.npy
*** Mean subtraction normalization with the mean: 
[[141.85724432 143.91169508 143.38139205 ... 129.0530303  126.67447917
  122.83309659]
 [143.50213068 145.11316288 144.81960227 ... 131.08143939 129.43868371
  126.30042614]
 [143.50331439 144.78622159 143.23910985 ... 130.74431818 129.42732008
  126.42637311]
 ...
 [126.16856061 126.46756629 125.69412879 ... 120.46117424 117.84232955
  112.87689394]
 [124.79403409 127.96543561 125.58120265 ... 119.5390625  115.97466856
  110.55965909]
 [123.47064394 127.4827178  126.66122159 ... 117.97632576 113.30752841
  108.23508523]]
('*** Image : \n', array([[115, 118, 117, ..., 138, 136, 130],
       [115, 117, 113, ..., 137, 134, 134],
       [125, 119, 117, ..., 137, 138, 135],
       ...,
       [185, 177, 181, ..., 100,  72,  51],
       [183, 178, 180, ...,  84,  93,  88],
       [176, 189, 183, ..., 101,  97,  75]], dtype=uint8))
('*** Mean Sub Image: \n', array([[-26.85724432, -25.91169508, -26.38139205, ...,   8.9469697 ,
          9.32552083,   7.16690341],
       [-28.50213068, -28.11316288, -31.81960227, ...,   5.91856061,
          4.56131629,   7.69957386],
       [-18.50331439, -25.78622159, -26.23910985, ...,   6.25568182,
          8.57267992,   8.57362689],
       ...,
       [ 58.83143939,  50.53243371,  55.30587121, ..., -20.46117424,
        -45.84232955, -61.87689394],
       [ 58.20596591,  50.03456439,  54.41879735, ..., -35.5390625 ,
        -22.97466856, -22.55965909],
       [ 52.52935606,  61.5172822 ,  56.33877841, ..., -16.97632576,
        -16.30752841, -33.23508523]]))
('*** Cell Count:\n', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 363, 335, 340, 325, 331, 506, 345, 0, 386, 0, 456, 394, 443, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
*** Done. Saved train data as LESS071018train_data-gray-er0.6-submean.npy
*** Done. Saved train class count array as train_classweight_gray-arraylabel-submean.npy




Processing train data... this may take a few minutes...
Reading cell data and returning as a dictionary...
*** Saved mean as... train_mean_gray-arraylabel-submean.npy
*** Done. Returning mean.
*** Mean subtraction normalization with the mean: 
[[137.11645387 138.24207166 138.32773888 ... 124.21324135 121.58896211
  117.63890033]
 [138.5488056  139.2095346  138.75607496 ... 125.46004942 123.48280478
  120.93729407]
 [138.59019769 139.13962109 137.83515239 ... 125.13447282 124.12716227
  122.00082372]
 ...
 [125.19831137 125.23404036 124.84071252 ... 119.12438221 116.7773888
  112.15012356]
 [123.91330313 126.5544687  124.58679984 ... 118.60203871 115.17740939
  110.00586903]
 [122.30508649 126.10286244 125.49649918 ... 116.70757825 112.29087727
  107.53902389]]
('*** Cell Count:\n', [31, 18, 18, 26, 13, 16, 27, 11, 17, 29, 19, 21, 28, 8, 20, 33, 17, 0, 1, 18, 14, 6, 3, 0, 0, 28, 4, 183, 363, 335, 340, 325, 331, 506, 345, 179, 386, 234, 456, 394, 443, 263, 96, 16, 6, 6, 14, 33, 9, 14, 12, 11, 9, 18, 10, 16, 16, 6, 14, 9, 10, 9, 13, 19, 13, 6, 2, 0, 0, 5, 5, 167, 22, 26, 28, 28, 27, 51, 61, 3, 156, 17, 16, 15, 76, 102, 278, 106, 82, 46, 28, 167, 37, 39, 0, 224, 15, 125, 89, 13, 46, 39, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 19, 134, 4, 60, 2, 34, 0, 15, 0, 0, 77, 48, 23, 56, 24, 21, 72, 30, 16, 61, 20, 257, 81, 58, 265, 0, 0, 0, 0, 0, 0, 0, 0, 206])
*** Done. Saved train data as train_data_gray-arraylabel-submean.npy
*** Done. Saved train class count array as train_classweight_gray-arraylabel-submean.npy
"""
