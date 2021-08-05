import ctypes
import ctypes.wintypes

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if ctypes.windll.user32.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
