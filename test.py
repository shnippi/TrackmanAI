import torch
from Trackmania_env import Trackmania_env
import numpy as np

env = Trackmania_env()
while True:
    env.get_cp()
    env.get_speed()




