from frameCellMap import FrameCellMap
from paths import DATA
import sys
# def findBottomX(x, numList):
#     """Given a number and a list of numbers, this finds the x smallest values in the number list, and reports
#     both the values, and their positions in the numList."""
#     topVals = [1e8] * x
#     topIndex = [None] * x
#     for i in range(len(numList)):
#         val = numList[i]
#
#         for j in range(x):
#             if topIndex[j] is None or val < topVals[j]:
#                 break
#         if val < topVals[x - 1]:
#             topIndex.insert(j, i)
#             topVals.insert(j, val)
#             topIndex.pop(-1)
#             topVals.pop(-1)
#     print(topVals, topIndex)
#
# findBottomX(3,[0.9,0.8,0.7,0.6])
# a = input()
# while not a == "a":
#     print("b")

# from pynput import keyboard
#
# def on_press(key):
#     if key == keyboard.Key.esc:
#         return False  # stop listener
#     try:
#         k = key.char  # single-char keys
#     except:
#         k = key.name  # other keys
#     if k in ['1', '2', 'left', 'right']:  # keys of interest
#         # self.keys.append(k)  # store it in global-like variable
#         print('Key pressed: ' + k)
#         return False  # stop listener; remove this if want more keys
#
# listener = keyboard.Listener(on_press=on_press)
# listener.start()  # start to listen on a separate thread
# listener.join()  # remove if main thread is polling self.keys
datafile = DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
labelMap = FrameCellMap(dataFile = datafile)
cell = labelMap.frameData[1234]['cell']
print(sys.getsizeof(type(cell)))

