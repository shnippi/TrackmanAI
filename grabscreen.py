# Done by Frannecklp


import pywintypes
import cv2
import numpy as np
# import win32gui, win32ui, win32con, win32api
import numpy as np
import cv2
from mss import mss
from PIL import Image


def grab_screen(region=None):
    mon = {'left': 0, 'top': 20, 'width': 790, 'height': 600}

    with mss() as sct:
        screenShot = sct.grab(mon)
        img = np.array(screenShot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
