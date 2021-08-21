import torch
from Trackmania_env import Trackmania_env
import numpy as np

import time

frametime = 0

env = Trackmania_env()
while True:
    frametime = time.time()
    env.get_cp()
    # print(time.time() - frametime)







