from networks import AE_net
import torch
import numpy as np
import cv2
from mss import mss
import time
from matplotlib import pyplot as plt


class Trackmania_env():

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_file_name = "E:/code/Python/Trackmania-RL/models/AE_1.model"
        self.net = AE_net()
        self.net.load_state_dict(torch.load(self.model_file_name, map_location=self.device))
        self.net.to(self.device)

    def reset(self):
        # gets game state (maybe resets race if possible with backspace)

        screen = self.grab_screen()
        screen = screen / 256
        screen = torch.from_numpy(screen)
        screen = torch.unsqueeze(screen, 0)  # add color dim
        screen = torch.unsqueeze(screen, 0)  # add batch dim
        screen = screen.to(self.device)
        screen = screen.float()

        # plt.imshow(screen[0][0].to("cpu"), "gray")
        # plt.show()
        # output = self.net(screen)
        # output = output.detach().to("cpu")
        # plt.imshow(output[0][0].to("cpu"), "gray")
        # plt.show()

        z = self.net.get_z(screen)
        z = torch.squeeze(z)
        z = z.detach().to("cpu")

        return z

    def step(self, action):
        # performs action and gets new gamestate
        pass

    def grab_screen(self, monitor_nr=1):

        if monitor_nr == 1:
            mon = {'left': 0, 'top': 250, 'width': 790, 'height': 350}

            with mss() as sct:
                img = np.array(sct.grab(mon))
                img = cv2.resize(img, (250, 250))
                # cv2.imshow('window', cv2.resize(img, (500, 500)))
                # key = cv2.waitKey(3000)
                # cv2.destroyAllWindows()
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


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
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
