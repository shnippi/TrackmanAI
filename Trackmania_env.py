from AE_networks import AE_net
import torch
import numpy as np
import cv2
from mss import mss
import time
from matplotlib import pyplot as plt
from directkeys import PressKey, ReleaseKey
import random

# key hexcodes
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
BACKSPACE = 0x0E


class Trackmania_env:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_file_name = "E:/code/Python/Trackmania-RL/models/AE_1.model"
        self.net = AE_net()
        self.net.load_state_dict(torch.load(self.model_file_name, map_location=self.device))
        self.net.to(self.device)

        self.observation_space = torch.tensor([256])
        self.action_space = TM_actionspace()

    def reset(self):
        time.sleep(4)
        # gets game state (maybe resets race if possible with backspace)
        PressKey(BACKSPACE)

        time.sleep(4)
        return np.array(self.get_state_rep())

    def step(self, action):
        # performs action and gets new gamestate
        if action[0] >= 0:
            PressKey(D)
            time.sleep(0.1)
            ReleaseKey(D)
        else:
            PressKey(A)
            time.sleep(0.1)
            ReleaseKey(A)

        if action[1] >= 0:
            PressKey(W)
            time.sleep(0.1)
            ReleaseKey(W)
        else:
            PressKey(S)
            time.sleep(0.1)
            ReleaseKey(S)

        z = np.array(self.get_state_rep())

        reward = 10

        return z, reward, False, None

    def get_state_rep(self, monitor_nr=1):

        if monitor_nr == 1:
            mon = {'left': 0, 'top': 250, 'width': 790, 'height': 350}

            with mss() as sct:
                img = np.array(sct.grab(mon))
                img = cv2.resize(img, (250, 250))
                # cv2.imshow('window', cv2.resize(img, (500, 500)))
                # key = cv2.waitKey(3000)
                # cv2.destroyAllWindows()
                screen = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        else:
            with mss() as sct:
                # Get information of monitor 2
                monitor_number = monitor_nr
                mon = sct.monitors[monitor_number]

                # The screen part to capture
                monitor = {
                    "top": mon["top"] + 300,  # 100px from the top
                    "left": mon["left"] + 400,  # 100px from the left
                    "width": 2200,
                    "height": 1200,
                    "mon": monitor_number,
                }

                # Grab the data_train
                img = np.array(sct.grab(monitor))
                img = cv2.resize(img, (250, 250))
                # cv2.imshow('window', cv2.resize(img, (500, 500)))
                # key = cv2.waitKey(3000)
                # cv2.destroyAllWindows()
                screen = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        screen = screen / 256
        screen = torch.from_numpy(screen)
        screen = torch.unsqueeze(screen, 0)  # add color dim
        screen = torch.unsqueeze(screen, 0)  # add batch dim
        screen = screen.to(self.device)
        screen = screen.float()

        plt.imshow(screen[0][0].to("cpu"), "gray")
        plt.show()
        output = self.net(screen)
        output = output.detach().to("cpu")
        plt.imshow(output[0][0].to("cpu"), "gray")
        plt.show()

        z = self.net.get_z(screen)
        z = torch.squeeze(z)
        z = z.detach().to("cpu")
        return z

    def random_action(self):
        return np.array([random.uniform(-1, 1), random.uniform(-1, 1)])


class TM_actionspace():
    def __init__(self):
        self.high = np.array([1., 1.])
        self.low = np.array([-1., -1.])
        self.shape = np.array([2])
