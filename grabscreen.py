# Done by Frannecklp

import io
import pywintypes
import cv2
import numpy as np
# import win32gui, win32ui, win32con, win32api
import numpy as np
import cv2
from mss import mss
from PIL import Image

def pil_frombytes(im):
    """ Efficient Pillow version. """
    return Image.frombytes('RGB', im.size, im.bgra, 'raw', 'BGRX').tobytes()


def grab_screen(region=None):
    mon = {'left': 0, 'top': 250, 'width': 790, 'height': 350}

    with mss() as sct:
        img = np.array(sct.grab(mon))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
