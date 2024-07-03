"""--------------------------------------------------------------------------------------------------------------------
Counts the number of images there are for every cell in the OLRI Map, considering the heading of the robot when the
picture was taken. It outputs a print statement in a table-like format, alongside a list of cells missing some frames
for one or more headings. It can also print how many times a given cell-heading combination appears in every file.
Made to work with the dataset format of 2022 and 2024.

Created: Summer 2024
Author: Oscar Reza B
--------------------------------------------------------------------------------------------------------------------"""
import os

# Define the path, for the Precision Towers, and read the files
dataPath2022 = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/"
filepath = os.listdir(dataPath2022)

# Set number of expected cells to 271 and expected headings to 8
cells, headings = (271, 8)
headingValues = [0, 45, 90, 135, 180, 225, 270, 315]


def countAppearances():
  """
  Returns an array with counts of the amount of image frames for all headings in a cell.
  """
  countArray = [[0 for x in range(headings)] for y in range(cells)]
  for file in filepath:
    appearances = 0
    if file.startswith("FrameData"):
      with open(dataPath2022 + file) as textFile:
        for line in textFile:
          words = line.split(" ")
          if len(words) > 1:
            cell = int(words[3])
            heading = int(words[4])
            if cell < cells and heading in headingValues:
              countArray[cell][headingValues.index(heading)] += 1

            # Looks for the txt filename that contains a heading-cell combination. This is manual input
            if cell == 218 and heading == 90:  # Change these values as needed
                appearances += 1
    # Checks if a given cell-heading combination appears too many times. Set to 50
    if appearances > 50:
      # Gets the name of the folder where the frame is at
      printable = str(file).replace("FrameDataReviewed", "").replace(".txt", "")
      print(f"Cell {cell} with heading {heading} appeared {str(appearances)} times in {printable} \n")
  return countArray


def createTable(appearanceCount):
  """
  Prints a table-like statement to display the output of appearanceCounter().
  """
  # Headers for the table
  headers = ['Cell'] + list(map(str, headingValues))  # Column names

  # Convert count data to a list-of-lists table
  datatable = [[str(i)] + list(map(str, appearanceCount[i])) for i in range(cells)]

  # Find the longest data value or header to be printed in each column
  widths = [max(len(value) for value in col) for col in zip(*(datatable + [headers]))]

  # Print heading followed by the data in datatable
  format_spec = ('{:{widths[0]}}  {:>{widths[1]}}  {:>{widths[2]}} {:>{widths[3]}} {:>{widths[4]}} {:>{widths[5]}} '
                 '{:>{widths[6]}} {:>{widths[7]}} {:>{widths[8]}}')
  print(format_spec.format(*headers, widths=widths))
  for fields in datatable:
    print(format_spec.format(*fields, widths=widths))
  print("")


def getZeroCount(appearanceCount):
  """
  Returns a list with the cell numbers that do not have any frames for one or more cell-heading combinations
  """
  zeroCounter = []
  for i in range(0, cells):
    for j in range(0, 8):
      if appearanceCount[i][j] == 0 and i != 18 and i != 19 and i != 114 and i != 115 and i != 116 and i != 117 and i != 152 and i not in zeroCount:
        zeroCounter.append(i)
  return zeroCounter


if __name__ == "__main__":
  count = countAppearances()
  createTable(count)
  zeroCount = getZeroCount(count)

  print("Cells missing some headings: " + str(zeroCount))
  print("Count: " + str(len(zeroCount)))
