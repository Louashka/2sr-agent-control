import pandas as pd
import numpy as np
import time
from datetime import datetime
from pynput import keyboard
import cv2
import PureThermal
import LUs
import mainController

portName = "COM4"
controller = mainController.Controller(portName)

s_current = [0, 0]

# create objects
thermal = PureThermal.PureThermal()
lu = LUs.LU()
lu.start()

flag = False
expData = []
imgList = []


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
                expData.append(
                    [timeStamp, t_current[0], q_current[3], q_current[4]])
                imgList.append(t_current[1])
                # for i in range(len(t_current[1])):
                #     if i == 0:
                #         expData.append(
                #             [timeStamp, t_current[0], q_current[3], q_current[4], t_current[1][0]])
                #     else:
                #         expData.append(([timeStamp, 0, 0, 0, t_current[1][i]]))
                # print(expData)
            break


def manualControl(v, s):
    global s_current
    # print(s_current)
    # print(s)
    counter = 0

    while True:
        q_current = lu.getCurrentConfig()
        if counter == 7:
            break
        counter += 1
        if q_current is not None:

            w = controller.wheelDrive(q_current, v, s)
            controller.moveRobotM(w, s)
            s_current = s

            break


def on_press(key):
    global expData
    flag = False
    show_image_flag = False

    move = True

    v = [0] * 5
    s = s_current

    lu_speed = 0.1

    if key == keyboard.KeyCode.from_char('q'):
        print('Recorded')
        flag = True

    if key == keyboard.KeyCode.from_char('e'):
        print('Stop and save')
        columnNames = ["time", "temperature", "k1", "k2"]
        df = pd.DataFrame(expData, columns=columnNames)
        df.to_csv('ExpData/exp_negative_curvature.csv', index=False)

        for i in range(len(imgList)):
            fileName = 'ExpData/TempPhotos/expNegativeCurvature_' + \
                str(i) + '.jpg'
            cv2.imwrite(fileName, imgList[i])

        lu.stop()
        thermal.stop()
        cv2.destroyAllWindows()

        move = False

    if key == keyboard.KeyCode.from_char('i'):
        show_image_flag = True

    if key == keyboard.KeyCode.from_char('s'):
        print("Segment 1 soft")
        s[0] = 1

    if key == keyboard.KeyCode.from_char('x'):
        print("Segment 1 rigid")
        s[0] = 0

    if key == keyboard.Key.up:
        print("forward")
        v[0] = lu_speed

    if key == keyboard.Key.down:
        print("backward")
        v[0] = -lu_speed

    if move:
        manualControl(v, s)

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
        # cv2.resizeWindow("Thermal", 300, 300)
        cv2.imshow("Thermal", img_col)
        k = cv2.waitKey(1)
        if k == 27:
            show_image_flag = False
            cv2.destroyAllWindows()
            break


def on_release(key):
    controller.moveRobotM(np.array([0, 0, 0, 0]), s_current)

    # cv2.destroyAllWindows()


# def data_collection():
#     global expData
#     q_current = lu.getCurrentConfig()  # q_current = [x, y, angle, ka, kb]
#     t_current = thermal.get_data()  # t_current = [temp_max, curveture]

#     timeStamp = datetime.now().strftime("%H:%M:%S")
#     for i in range(len(t_current[1])):
#         if i == 1:
#             expData.append(
#                 [timeStamp, t_current[0], q_current[3], q_current[4], t_current[2]])
#         else:
#             expData.append(([timeStamp, 0, 0, 0, 0, t_current[1][i]]))
#     print(expData)


while True:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
