import pandas as pd
import numpy as np
import time
from datetime import datetime
from pynput import keyboard
import cv2
import PureThermal
import LUs

# create objects
thermal = PureThermal.PureThermal()
lu = LUs.LU()
lu.start()

flag = False
expData = []

START_DATA_FLOW = {keyboard.KeyCode.from_char('q')}
STOP_AND_SAVE = {keyboard.KeyCode.from_char('s')}
SHOW_IMAGE = {keyboard.KeyCode.from_char('i')}
# The currently active modifiers
current_keys = set()


def get_data():
    global expData
    counter = 0
    while True:
        q_current = lu.getCurrentConfig()  # q_current = [x, y, angle, ka, kb]
        t_current = thermal.get_data()  # t_current = [temp_max, curveture]
        if counter == 7:
            break
        counter += 1
        if q_current is not None:
            # print(q_current)
            if t_current is not None:
                # print(t_current)
                timeStamp = datetime.now().strftime("%H:%M:%S")
                for i in range(len(t_current[1])):
                    if i == 0:
                        expData.append(
                            [timeStamp, t_current[0], q_current[3], q_current[4], t_current[1][0]])
                    else:
                        expData.append(([timeStamp, 0, 0, 0, t_current[1][i]]))
                print(expData)
                break


def on_press(key):
    global flag, expData
    show_image_flag = False

    if key in START_DATA_FLOW:
        current_keys.add(key)
        if key.char == 'q':
            print('Recorded')
            flag = True

    if key in STOP_AND_SAVE:
        current_keys.add(key)
        if key.char == 's':
            print('Stop and save')
            flag = False
            columnNames = ["time", "temperature", "ka*", "kb*", "ka/b"]
            df = pd.DataFrame(expData, columns=columnNames)
            lu.stop()
            thermal.stop()
            print("save")
            df.to_csv('expData.csv', index=False)

    if key in SHOW_IMAGE:
        current_keys.add(key)
        if key.char == 'i':
            show_image_flag = True

    if flag:
        get_data()

    while show_image_flag:
        image = thermal.grab_image(thermal.camera)
        thermal.show_image(image)
        image = np.array(image)
        img = np.array(image)
        img_new = img - img.min()
        img_deno: object = img.max() - img.min()
        img = 255 * (img_new / img_deno)
        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)
        # show image
        cv2.namedWindow("Thermal", 0)
        cv2.resizeWindow("Thermal", 300, 300)
        cv2.imshow("Thermal", img_col)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            show_image_flag = False
            break


def on_release(key):
    global current_keys
    try:
        current_keys.remove(key)
        # controller.moveRobotM(np.aqrray([0, 0, 0, 0]), s_current, False, False)
    except KeyError:
        pass
    cv2.destroyAllWindows()


def data_collection():
    global expData
    q_current = lu.getCurrentConfig()  # q_current = [x, y, angle, ka, kb]
    t_current = thermal.get_data()  # t_current = [temp_max, curveture]

    timeStamp = datetime.now().strftime("%H:%M:%S")
    for i in range(len(t_current[1])):
        if i == 1:
            expData.append(
                [timeStamp, t_current[0], q_current[3], q_current[4], t_current[2]])
        else:
            expData.append(([timeStamp, 0, 0, 0, 0, t_current[1][i]]))
    print(expData)


while True:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
