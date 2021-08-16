from AE_networks import AE_net
import torch
import numpy as np
import cv2
from mss import mss
import time
from matplotlib import pyplot as plt
from directkeys import PressKey, ReleaseKey
import random
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

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

        self.observation_space = torch.zeros(256)
        self.action_space = TM_actionspace()

        # save last measured checkpoint and speed such that if the OCR fails we just take last measured
        self.cp = [""]
        self.first_cp_predict_counter = 0
        self.speed = 0
        self.time = ""
        self.stuck_counter = 0

    def reset(self):
        # reset
        PressKey(BACKSPACE)
        ReleaseKey(BACKSPACE)

        self.cp = [""]
        self.first_cp_predict_counter = 0
        self.speed = 0
        self.time = ""
        self.stuck_counter = 0

        # print("RESETTING...")
        for i in list(range(2))[::-1]:
            time.sleep(1)

        return np.array(self.get_state_rep())

    def step(self, action):

        done = False

        # TODO: instead of sleep do sth better, such that no idle time of code
        ReleaseKey(D)
        PressKey(A)
        ReleaseKey(W)
        ReleaseKey(S)

        # performs action and gets new gamestate
        if action[0] >= 0:
            PressKey(D)
        else:
            PressKey(A)

        # TODO: currently only going forward
        if action[1] >= 0:
            PressKey(W)

        else:
            PressKey(S)

        z = np.array(self.get_state_rep())

        speed = self.get_speed()
        cp, cp_reached = self.get_cp()

        reward = (speed / 150) ** 2 - 0.15
        if cp_reached:
            reward += 50

        if speed == 0:
            self.stuck_counter += 1
            if self.stuck_counter > 30:
                self.stuck_counter = 0
                done = True
                reward = -100

        # print("speed: " + str(speed) + " ; cp: " + str(cp) + " ; reward: " + str(reward))

        return z, reward, done, None

    def get_state_rep(self, monitor_nr=1):

        if monitor_nr == 1:
            mon = {'left': 0, 'top': 250, 'width': 790, 'height': 350}

            with mss() as sct:
                img = np.array(sct.grab(mon))
                img = cv2.resize(img, (250, 250))
                # cv2.imshow('window', cv2.resize(img, (500, 500)))
                # key = cv2.waitKey(0)
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

                # Grab the data_train_250
                img = np.array(sct.grab(monitor))
                img = cv2.resize(img, (250, 250))
                # cv2.imshow('window', cv2.resize(img, (500, 500)))
                # key = cv2.waitKey(3000)
                screen = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

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

    # TODO: doest recognize 1
    # TODO: better checkpoint safetyguards (can only increment by 1 etc)
    def get_cp(self):

        cp_reached = False

        mon = {'left': 350, 'top': 550, 'width': 100, 'height': 30}

        with mss() as sct:
            img = np.array(sct.grab(mon))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img = (img > 250) * img  # only take the pure white part of image (where the values are displayed)
            img = cv2.resize(img, (200, 60))

            cp_reached = False

            # first = img[:, 40:85]
            # first = cv2.resize(first, (28, 28))
            # string = pytesseract.image_to_string(first)
            # print(string)
            # cv2.imshow("result",first)
            # cv2.waitKey(0)

            # cv2.imshow("result",img)
            # cv2.waitKey(0)

            string = pytesseract.image_to_string(img)
            cp = ""
            for i in string:
                if i.isdigit() or i == "/":
                    cp += i

            # print(cp.split("/"))
            # print(self.cp)
            cp = cp.split("/")
            # print(cp)

            # check if "/" got recognized as a number
            if len(cp) == 1:
                cp = [""]

            # if the ocr picked sth up
            if cp != [""] and cp[0] != "" and cp[1] != "":

                if self.cp == [""]:
                    self.cp = cp

                # checkpoint reached, only update checkpoint value upon reaching checkpoint.
                if cp[0] != self.cp[0] and cp[0] != "0" and int(cp[0]) % int(cp[1]) == (int(self.cp[0]) + 1) % int(
                        cp[1]):
                    print("checkpoint!")
                    cp_reached = True
                    self.cp = cp

                return self.cp, cp_reached

            # try to predict if it hit first cp
            elif cp == [""] and self.cp[0] == "0":
                self.first_cp_predict_counter += 1

                if self.first_cp_predict_counter >= 10:
                    print("checkpoint!")
                    cp_reached = True
                    self.cp[0] = "1"

                    return self.cp, cp_reached

            return self.cp, cp_reached

    def get_speed(self):

        mon = {'left': 550, 'top': 260, 'width': 60, 'height': 30}

        with mss() as sct:
            img = np.array(sct.grab(mon))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img = (img > 150) * img  # only take the pure white part of image (where the values are displayed)
            img = cv2.resize(img, (240, 120))

            # cv2.imshow("result",img)
            # cv2.waitKey(0)

            string = pytesseract.image_to_string(img)
            speed = ""
            for i in string:
                if i.isdigit():
                    speed += i
            # print(speed)
            # print(self.speed)

            # check if OCR worked, else take old value
            if speed:
                self.speed = int(speed)
                return self.speed
            else:
                self.speed = max(self.speed - 5, 0)
                return self.speed

    def get_time(self):

        mon = {'left': 350, 'top': 575, 'width': 100, 'height': 40}

        with mss() as sct:
            img = np.array(sct.grab(mon))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img = (img > 250) * img  # only take the pure white part of image (where the values are displayed)
            img = cv2.resize(img, (140, 60))

            # cv2.imshow("result",img)
            # cv2.waitKey(0)

            string = pytesseract.image_to_string(img)
            print(string)
            time = ""
            for i in string:
                if i.isdigit():
                    time += i

            print(self.speed)

            # check if OCR worked, else take old value
            if time:
                self.speed = int(time)
                return self.speed
            else:
                self.speed = max(self.speed - 5, 0)
                return self.speed

    def random_action(self):
        return np.array([random.uniform(-1, 1), random.uniform(-1, 1)])


class TM_actionspace():
    def __init__(self):
        self.high = np.array([1., 1.])
        self.low = np.array([-1., -1.])
        self.shape = np.array([2])
