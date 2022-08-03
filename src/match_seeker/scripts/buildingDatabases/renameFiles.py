"""
renameFiles.py
Utility script for renaming filenames with colons in a given directory
"""
import os
from src.match_seeker.scripts.olri_classifier.paths import data2022rename


def removeColons(folderPath)\
    :
    folderPath = folderPath + '/'
    for file in os.listdir(folderPath):
        fileName, fileExt = os.path.splitext(file)
        if ':' in fileName:
            newFileName = fileName.replace(':', '')
            newName = newFileName + fileExt
            os.rename(folderPath + file, folderPath + newName)


if __name__ == '__main__':
    True
    # removeColons(data2022rename + '20220715-1322frames')
    # removeColons(data2022rename + '20220715-1613frames')
    # removeColons(data2022rename + '20220718-1438frames')
    # removeColons(data2022rename + '20220721-1408frames')
    # removeColons(data2022rename + '20220722-1357frames')
    # removeColons(data2022rename)


