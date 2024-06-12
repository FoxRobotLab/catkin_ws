import cv2
import os

def imageChecker(directory_path):
    for root, _, files in os.walk(directory_path):
        files.sort()
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is None:
                    print("Could not open or find the image: {}".format(image_path))
                    continue
                print("Image ")
                cv2.imshow('Image', image)
                cv2.waitKey(500)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    imageChecker("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/DATA/FrameData")