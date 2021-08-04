from pynput.keyboard import Key, Listener
import time
from threading import Timer

keys = []


# TODO: improve key detection
def on_press(key):
    try:
        keys.append(key.char)
    except AttributeError:
        pass


def key_check():
    with Listener(on_press=on_press) as l:
        Timer(0.1, l.stop).start()
        l.join()

    return list(set(keys))
