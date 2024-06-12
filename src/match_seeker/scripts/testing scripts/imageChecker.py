import cv2
import os
import re


def extractSpecificNumbers(line):
    # Find all sequences of 1 to 3 digits in the line
    numbers = re.findall(r'\b\d{1,3}\b', line)
    if len(numbers) >= 6:
        return numbers[4], numbers[5]
    return None, None


def findTextFile(directoryName, textFiles):
    # Extract the date portion from the directory name
    datePortion = directoryName.split('-')[0]
    for textFile in textFiles:
        if datePortion in textFile:
            return textFile
    return None


def imageChecker(baseDirectory):
    # Go one level higher than the DATA directory
    grandparentDirectory = os.path.dirname(os.path.dirname(baseDirectory))
    textFiles = [f for f in os.listdir(grandparentDirectory) if f.lower().endswith('.txt')]
    endList = []

    for root, directories, _ in os.walk(baseDirectory):
        for directory in directories:
            # Full path to the image directory
            directoryPath = os.path.join(root, directory)

            # Find the corresponding text file for the current directory
            textFileName = findTextFile(directory, textFiles)
            if not textFileName:
                print("No matching text file found for directory: {}".format(directory))
                endList.append(directory)
                continue

            textFilePath = os.path.join(grandparentDirectory, textFileName)
            print("Using text file: {} for directory: {}".format(textFilePath, directoryPath))

            for _, _, files in os.walk(directoryPath):
                files.sort()
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        imagePath = os.path.join(directoryPath, file)
                        image = cv2.imread(imagePath)
                        if image is None:
                            print("Could not open or find the image: {}".format(imagePath))
                            continue

                        # Extract portion of the file name (assuming it's before the first underscore)
                        filePortion = file.split('_')[0]

                        # Open and search the text file
                        with open(textFilePath, 'r') as textFile:
                            lines = textFile.readlines()
                            for line in lines:
                                if filePortion in line:
                                    num1, num2 = extractSpecificNumbers(line)
                                    if num1 and num2:
                                        print("Cell: {} Heading: {}".format(num1, num2))
                                    else:
                                        print(
                                            "Found matching line for {} but could not extract the specified numbers".format(
                                                file))
                                    break

                        # Display the image
                        cv2.imshow('Image', image)
                        cv2.waitKey(1)
            cv2.destroyAllWindows()
    print endList


if __name__ == "__main__":
    imageChecker("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/DATA/FrameData")

"""Missing folders:
20220705-1446frames
20220705-1616frames
20220713-1136frames
20220713-1411frames
20220713-1548frames
20220715-1322frames
20220715-1613frames
20220718-1438frames
20220721-1408frames
20220722-1357frames
20220727-1510frames
20220728-1423frames
20220728-1445frames
20220729-1620frames
20220801-1422frames
20220802-1043frames
20220802-1521frames
20220803-1047frames
20220803-1135frames
"""
