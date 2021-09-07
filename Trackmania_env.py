from networks import AE_net, VanillaVAE, LeNet_plus_plus
import torch
import numpy as np
import cv2
from mss import mss
import time
from matplotlib import pyplot as plt
from directkeys import PressKey, ReleaseKey
import random
import pytesseract
import sys

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# key hexcodes
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
BACKSPACE = 0x0E
ENTER = 0x1C

img_dim = 64
z_dim = 33


class Trackmania_env:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.VAE_model_file_name = "../models/VAE_64_100eps_vanilla_recon_game.model"
        self.VAE_net = VanillaVAE()
        self.VAE_net.load_state_dict(torch.load(self.VAE_model_file_name, map_location=self.device))
        self.VAE_net.to(self.device)

        self.MNIST_model_file_name = "../models/MNIST_classifier.model"
        self.MNIST_net = LeNet_plus_plus()
        self.MNIST_net.load_state_dict(torch.load(self.MNIST_model_file_name, map_location=self.device))
        self.MNIST_net.to(self.device)

        self.observation_space = torch.zeros(z_dim)
        self.action_space = TM_actionspace()

        # save last measured checkpoint and speed such that if the OCR fails we just take last measured
        self.cp = [""]
        self.cp_predict_counter = 0
        self.cp_predict = [""]
        self.speed = 0
        self.start_time = 0
        self.stuck_counter = 0
        self.course_done_counter = 0

        self.cp1_numbers, self.cp2_numbers = self.load_cp_numbers()
        self.minus = np.load('E:/code/Python/Trackmania-RL/data/checkpoint_digits/minus.npy')

        self.update_time = 0

    def reset(self):
        # reset
        PressKey(BACKSPACE)
        ReleaseKey(BACKSPACE)

        self.cp = [""]
        self.cp_predict_counter = 0
        self.speed = 0
        self.start_time = time.time()
        self.stuck_counter = 0
        self.course_done_counter = 0

        # print("RESETTING...")
        for i in list(range(2))[::-1]:
            time.sleep(1)

        return np.array(self.get_state_rep())

    # TODO: maybe do sequence of screenshots for state
    # TODO: watch the vid and look what he did
    def step(self, action):

        done = False

        ReleaseKey(D)
        PressKey(A)
        ReleaseKey(W)
        ReleaseKey(S)

        # LEFT / STRAIGHT / RIGHT
        if action[0] >= 0.5:
            PressKey(D)
        elif action[0] <= -0.5:
            PressKey(A)

        # ACCELERATE / IDLE / BREAK
        if action[1] >= 0:
            PressKey(W)
        # elif action[1] <= -0.5:
        #     PressKey(S)

        else:
            pass

        # TODO: maybe train the model with printed digits instead of handwritten
        speed = self.get_speed()
        cp, cp_reached = self.get_cp()

        z = np.array(self.get_state_rep())
        # print(z)

        reward = (speed / 150) ** 2 - 0.15
        if cp_reached:
            reward += 20
            # print("Checkpoint! :D")

        if speed < 5:
            self.stuck_counter += 1
            if self.stuck_counter > 100:
                self.stuck_counter = 0
                done = True
                reward = min(- 50 + time.time() - self.start_time, 0)
                # print("oooopsie woopsie stuckie wuckie")
        else:
            self.stuck_counter = 0

        if self.course_done_counter > 50:
            print("COURSE COMPLETED! :D")
            self.course_done_counter = 0
            PressKey(ENTER)
            time.sleep(0.1)
            ReleaseKey(ENTER)
            done = True
            reward = 10000000 / ((time.time() - self.start_time) ** 2)

        # print("speed: " + str(speed) + " ; cp: " + str(cp) + " ; reward: " + str(reward))
        # print(1/(time.time()-self.update_time))
        # self.update_time = time.time()

        return z, reward, done, None

    def get_state_rep(self):

        mon = {'left': 0, 'top': 250, 'width': 790, 'height': 350}

        with mss() as sct:
            img = np.array(sct.grab(mon))
            img = cv2.resize(img, (img_dim, img_dim))
            # cv2.imshow('window', cv2.resize(img, (500, 500)))
            # key = cv2.waitKey(0)
            screen = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # check if course done (if done black bars appear on top and bottom)
        bottom_bar = screen[60:, 10:40]
        if np.amax(bottom_bar) == 0:
            self.course_done_counter += 1
        else:
            self.course_done_counter = 0

        # prepare the image for the VAE
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

        # self.show_reconstruction(screen)

        z = self.VAE_net.get_z(screen)
        z = torch.squeeze(z)
        z = z.detach().to("cpu")

        # add normalized speed to state representation
        z_speed = np.interp(self.speed, [0, 300], [-1, 1])
        z = torch.cat((z, torch.tensor([z_speed])))

        return z

    # TODO: train MNIST on own dataset
    # TODO: better checkpoint safetyguards (can only increment by 1 etc)
    def get_cp(self):

        cp_reached = False

        mon = {'left': 370, 'top': 550, 'width': 60, 'height': 28}
        cp1 = np.zeros((28, 28))
        cp2 = np.zeros((28, 28))
        cp_list = []
        cp = []

        with mss() as sct:

            img = np.array(sct.grab(mon))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img = (img > 250) * img  # only take the pure white part of image (where the values are displayed)

            # cv2.imshow("result", img[:, 37:])
            # cv2.waitKey(0)

            cp1[:, 4:24] = img[:, :20]
            cp_list.append(cp1)
            cp2[:, 3:26] = img[:, 37:]
            cp_list.append(cp2)

            # using mnist model
            # for digit in cp_list:
            #     if np.amax(digit) > 240:
            #         digit = torch.from_numpy(digit).float()
            #         digit = digit / 256
            #         digit = torch.unsqueeze(digit, 0)
            #         digit = torch.unsqueeze(digit, 0)
            #         digit = digit.to(self.device)
            #         pred = self.MNIST_net(digit)
            #         pred = pred.argmax(1).item()
            #
            #         # TODO: this is yikes
            #         if pred == 7:
            #             pred = 1
            #         # print(pred.argmax(1))dw
            #         cp.append(str(pred))

            # using pixel differences
            pred = ""
            sum_diff = 100000
            for index in range(len(self.cp1_numbers)):
                diff = np.sum(np.absolute(cp1 - self.cp1_numbers[index]))
                if diff < sum_diff:
                    sum_diff = diff
                    pred = index
            cp.append(str(pred))

            pred = ""
            sum_diff = 100000
            for index in range(len(self.cp2_numbers)):
                diff = np.sum(np.absolute(cp2 - self.cp2_numbers[index]))
                if diff < sum_diff:
                    sum_diff = diff
                    pred = index
            cp.append(str(pred))

            # print(cp)
            # print(self.cp_predict_counter)

            # TODO: 0 doesnt get updated here
            if cp[0] != "" and cp[0] != "0" and cp != self.cp:
                self.cp_predict_counter += 1
                if self.cp_predict_counter > 10:
                    # print("checkpoint!")
                    self.cp_predict_counter = 0
                    cp_reached = True
                    self.cp = cp

            return self.cp, cp_reached

    def get_speed(self):

        mon = {'left': 550, 'top': 260, 'width': 60, 'height': 28}
        digits = []
        speed = ""
        digit1 = np.zeros((28, 28))
        digit2 = np.zeros((28, 28))
        digit3 = np.zeros((28, 28))

        with mss() as sct:
            img = np.array(sct.grab(mon))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img = (img > 150) * img  # only take the pure white part of image (where the values are displayed)

            diff = np.sum(np.absolute(img[:, :10] - self.minus))
            if diff < 3000:
                self.speed = 0
                return self.speed

            # pad the images to 28x28
            digit1[:, 5:22] = img[:, :17]
            digits.append(digit1)
            digit2[:, 7:20] = img[:, 17:30]
            digits.append(digit2)
            digit3[:, 8:20] = img[:, 30:42]
            digits.append(digit3)

            # read speed with MNIST model
            for digit in digits:
                if np.amax(digit) > 240:
                    digit = torch.from_numpy(digit).float()
                    digit = digit / 256
                    digit = torch.unsqueeze(digit, 0)
                    digit = torch.unsqueeze(digit, 0)
                    digit = digit.to(self.device)
                    pred = self.MNIST_net(digit)
                    pred = pred.argmax(1).item()

                    # TODO: this is yikes
                    if pred == 7:
                        pred = 1
                    # print(pred.argmax(1))
                    speed += str(pred)

            if speed:
                self.speed = int(speed)

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

            # check if OCR worked, else take old value
            if time:
                self.speed = int(time)
                return self.speed
            else:
                self.speed = max(self.speed - 5, 0)
                return self.speed

    def show_reconstruction(self, screen):

        recon = self.VAE_net.generate(screen)
        recon = np.array(recon.detach().to("cpu"))
        cv2.imshow('window', cv2.resize(recon[0][0], (500, 500)))
        cv2.waitKey(10)

    def random_action(self):
        return np.array([random.uniform(-1, 1), random.uniform(-1, 1)])

    def load_cp_numbers(self):
        cp1_numbers = [np.load("../data/checkpoint_digits/zero1.npy"), np.load("../data/checkpoint_digits/one1.npy"),
                       np.load("../data/checkpoint_digits/two1.npy"), np.load("../data/checkpoint_digits/three1.npy"),
                       np.load("../data/checkpoint_digits/four1.npy"), np.load("../data/checkpoint_digits/five1.npy"),
                       np.load("../data/checkpoint_digits/six1.npy"), np.load("../data/checkpoint_digits/seven1.npy"),
                       np.load("../data/checkpoint_digits/eight1.npy"), np.load("../data/checkpoint_digits/nine1.npy")]

        cp2_numbers = [np.load("../data/checkpoint_digits/zero2.npy"), np.load("../data/checkpoint_digits/one2.npy"),
                       np.load("../data/checkpoint_digits/two2.npy"), np.load("../data/checkpoint_digits/three2.npy"),
                       np.load("../data/checkpoint_digits/four2.npy"), np.load("../data/checkpoint_digits/five2.npy"),
                       np.load("../data/checkpoint_digits/six2.npy"), np.load("../data/checkpoint_digits/seven2.npy"),
                       np.load("../data/checkpoint_digits/eight2.npy"), np.load("../data/checkpoint_digits/nine2.npy")]

        return cp1_numbers, cp2_numbers


class TM_actionspace():
    def __init__(self):
        self.high = np.array([1., 1.])
        self.low = np.array([-1., -1.])
        self.shape = np.array([2])
