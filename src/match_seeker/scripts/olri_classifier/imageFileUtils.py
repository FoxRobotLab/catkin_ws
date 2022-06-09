


def makeFilename(path, fileNum):
    """Makes a filename for reading or writing image files"""
    formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
    name = formStr.format(path, 'frames', fileNum, "jpg")
    return name


def getImageFilenames(path):
    """Read filenames in folder, and keep those that end with jpg or png  (from copyMarkedFiles.py)"""
    filenames = os.listdir(path)
    keepers = []
    for name in filenames:
        if name.endswith("jpg") or name.endswith("png"):
            keepers.append(name)
    return keepers


def extractNum(fileString):
    """Finds sequence of digits"""
    numStr = ""
    foundDigits = False
    for c in fileString:
        if c in '0123456789':
            foundDigits = True
            numStr += c
        elif foundDigits:
            break
    if numStr != "":
        return int(numStr)
    else:
        return -1

