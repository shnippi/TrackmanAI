import torch
from Trackmania_env import Trackmania_env

env = Trackmania_env()
# env.reset()
# env.step([0.1,0.1])
print(env.random_action())