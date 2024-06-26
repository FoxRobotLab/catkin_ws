"""
This script counts the number of images there are for every cell in the OLRI Map,
considering the heading of the robot when the picture was taken.
"""
import os

# Define the path and read the files
currentPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/"
filepath = os.listdir(currentPath)

cells, headings = (271, 9)
headingValues = [0, 45, 90, 135, 180, 225, 270, 315, 360]


"""
Counts the amount of images for all headings in a cell.
"""
def counter():
  arr = [[0 for x in range(headings)] for y in range(cells)]
  for file in filepath:
    appearances = 0
    if file.startswith("FrameData"):
      with open(currentPath + file) as textFile:
        for line in textFile:
          words = line.split(" ")
          if len(words) > 1:
            cell = int(words[3])
            heading = int(words[4])
            if cell < cells and heading in headingValues:
              arr[cell][headingValues.index(heading)] += 1
            # Looks for the txt filename that contains a heading-cell combination that is over-represented
            if cell == 218 and heading == 90:  # Change these values as needed
                appearances += 1
    # Prints the number of appearances in a folder to help locate it
    if appearances > 30:
      printable = str(file).replace("FrameDataReviewed", "").replace(".txt", "")
      print("Appeared " + str(appearances) + " times in " + printable)
  return arr


count = counter()
zeroCount = []

# Headers for the table
headers = ['Cell'] + list(map(str, headingValues))  # Column names

# Convert count data to a list-of-lists table
datatable = [[str(i)] + list(map(str, count[i])) for i in range(cells)]

# Find the longest data value or header to be printed in each column
widths = [max(len(value) for value in col) for col in zip(*(datatable + [headers]))]

# Print heading followed by the data in datatable
format_spec = ('{:{widths[0]}}  {:>{widths[1]}}  {:>{widths[2]}} {:>{widths[3]}} {:>{widths[4]}} {:>{widths[5]}} '
              '{:>{widths[6]}} {:>{widths[7]}} {:>{widths[8]}} {:>{widths[9]}}')
print(format_spec.format(*headers, widths=widths))
for fields in datatable:
  print(format_spec.format(*fields, widths=widths))
print("")

# Print the cells that have no images taken in any heading
for i in range(0, cells):
  for j in range(0, 9):
    if count[i][j] == 0 and i != 18 and i != 19 and i != 114 and i != 115 and i != 116 and i != 117 and i != 152 and i not in zeroCount:
      zeroCount.append(i)

print("Cells missing some headings: " + str(zeroCount))
print("Count: " + str(len(zeroCount)))

