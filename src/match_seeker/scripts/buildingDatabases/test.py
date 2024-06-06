#!/usr/bin/env python2.7

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

import cv2
import numpy as np


blank_image = 255 * np.ones(shape=[300 , 1200 , 3], dtype=np.uint8)

firstline = "a: left turn 360 || d: right turn 360"
secondline = "w: move forward indefinitely || x: move backward indefinitely"
thirdline = "z: keep hitting for moving left || c: keep hitting for moving right"
fourthline = "q: quit || s: stop"

turtle_image_text = cv2.putText(img=blank_image, text=firstline, org=(50, 50), fontScale=1,
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0,0,0))
turtle_image_text = cv2.putText(img=turtle_image_text, text=secondline, org=(50, 80), fontScale=1,
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 0))
turtle_image_text = cv2.putText(img=turtle_image_text, text=thirdline, org=(50, 110), fontScale=1,
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 0))
turtle_image_text = cv2.putText(img=turtle_image_text, text=fourthline, org=(50, 140), fontScale=1,
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 0))

cv2.imshow("Turtle Image", turtle_image_text)

cv2.waitKey(10)

cv2.destroyAllWindows()
