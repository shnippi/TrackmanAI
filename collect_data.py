import numpy as np
import cv2
import time
from getkeys import key_check
import os
from mss import mss

# TODO: trackmania settings on 800x600

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

starting_value = 0


# starting from the file we left off
while True:
    if os.path.isfile('E:/code/Python/Trackmania-RL/data_train/train/training_data-{}.npy'.format(starting_value)):
        starting_value += 1
    else:
        print('Starting from file: ', starting_value)
        break

def grab_screen(monitor_nr=1):

    if monitor_nr == 1:
        mon = {'left': 0, 'top': 250, 'width': 790, 'height': 350}
        with mss() as sct:
            img = np.array(sct.grab(mon))
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
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


def main(starting_value):
    training_data = []
    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    print('STARTING!!!')
    while True:

        if not paused:
            screen = grab_screen(monitor_nr=1)
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (250, 250))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            training_data.append(screen)

            # # run a color convert:
            # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            # keys = key_check()
            # output = keys_to_output(keys)
            # training_data.append(screen)

            cv2.imshow('window', cv2.resize(screen, (500, 500)))
            cv2.waitKey(0)

            # display recorded stream
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if len(training_data) % 100 == 0:
                if len(training_data) == 500:
                    file_name = 'E:/code/Python/Trackmania-RL/data_train/train/training_data-{}.npy'.format(starting_value)
                    np.save(file_name, training_data)
                    print('SAVED file {}'.format(starting_value))
                    training_data = []
                    starting_value += 1

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


def keys_to_output(keys):
    """
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    """
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


main(starting_value)
