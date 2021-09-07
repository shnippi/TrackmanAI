import torch
from Trackmania_env import Trackmania_env
import numpy as np
import time
from directkeys import PressKey, ReleaseKey


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
BACKSPACE = 0x0E
ENTER = 0x1C

frametime = 0

env = Trackmania_env()
while True:
    # print(env.get_cp())
    print(env.get_speed())







