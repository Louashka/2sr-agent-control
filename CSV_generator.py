import pandas as pd
import numpy as np
import time
from datetime import datetime
from pynput import keyboard

flag = False
expData = []

START_DATA_FLOW = {keyboard.KeyCode.from_char('q')}
STOP_AND_SAVE = {keyboard.KeyCode.from_char('s')}
# The currently active modifiers
current_keys = set()


def on_press(key):
    global flag
    if key in START_DATA_FLOW:
        current_keys.add(key)
        if all(k in current_keys for k in START_DATA_FLOW):
            print('Start recording')
            flag = True
            print("flag in function", flag)

    if key in STOP_AND_SAVE:
        current_keys.add(key)
        if all(k in current_keys for k in STOP_AND_SAVE):
            print('Stop and save')
            flag = False
            print("flag in function", flag)
            columnNames = ["time", "temperature", "ka*", "kb*", "ka", "kb"]
            df = pd.DataFrame(expData, columns=columnNames)
            print("save")
            df.to_csv('expData.csv', index=False)

    if flag:
        data_collection()


def on_release(key):
    global current_keys
    try:
        current_keys.remove(key)
        # controller.moveRobotM(np.aqrray([0, 0, 0, 0]), s_current, False, False)
    except KeyError:
        pass


def data_collection():
    global expData
    timeStamp = datetime.now().strftime("%H:%M:%S")
    expData.append([timeStamp, "bla", "bla", 0, 0, 0])
    print(expData)


while True:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    data_collection()








