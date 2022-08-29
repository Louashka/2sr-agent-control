import numpy as np
import pandas as pd
import time
from datetime import datetime
from pynput import keyboard
from threading import Thread
import mainController
import LUs

expData = []
# w = np.array([[10, 0, 0, 0]])
# s = [[1, 0]]
# flag = [False]

lu = LUs.LU()
lu.start()

q_current = lu.getCurrentConfig()
q_target = [0.28, 0.14, -0.4, -29, 27]
s_current = [0, 0]

portName = "COM4"
controller = mainController.Controller(portName)
# controller.moveRobot(w, s, [True])


def unitsStabilityExpLoop():
    while True:
        # start the motion
        centers, rotation = lu.DetectArucoPose()
        timeStamp = datetime.now().strftime("%H:%M:%S")
        expData.append([timeStamp, centers[2, 0], centers[2, 1], rotation[2],
                        centers[1, 0], centers[1, 1], rotation[1]])

        controller.moveRobot(w, s, flag)

        # time.sleep(0.05)


def mainExperiment():
    global q_current, s_current

    dist = np.linalg.norm(q_current[:3] - q_target[:3])

    while dist > 10**(-2):

        q_current = lu.getCurrentConfig()
        if q_current is None:
            continue
        print("Current config: ", q_current)
        config = controller.motionPlanner(q_current, q_target, s_current)
        w = controller.wheelDrive(config[0], config[1], config[2])

        controller.moveRobot(w, config[2], config[3])

        q_current = lu.getCurrentConfig()
        s_current = config[2]

        if q_current is None:
            continue

        # print(q_current)
        error = np.linalg.norm(config[0] - q_current)
        dist = np.linalg.norm(q_current[:3] - q_target[:3])
        timeStamp = datetime.now().strftime("%H:%M:%S")

        expData.append([config[0][0], config[0][1], config[0][2], config[0][3], config[0][4], config[1][0], config[1][1], config[1][2], config[1][3], config[1][4], q_current[0], q_current[1], q_current[2],
                        q_current[3], q_current[4], s_current[0], s_current[1],  timeStamp])

    w = np.array([[0, 0, 0, 0]])
    s = [0, 0]
    flag = True
    controller.moveRobot(w, s, flag)

    columnNames = ["x_model", "y_model", "angle_model", "k1_model", "k2_model", "v1", "v2", "u0", "v0", "r0", "x", "y", "angle", "k1", "k2",
                   "s1", "s2", "time"]
    df = pd.DataFrame(expData, columns=columnNames)
    print("save")
    df.to_csv('ExpData/mainExperiment1.csv', index=False)

    controller.closeConnection()


def on_press(key):
    global w, s, flag

    if key.char == "q":
        print('Aborted')

        w = np.array([[0, 0, 0, 0]])
        s = [0, 0]
        flag = True
        controller.moveRobot(w, s, flag)

        controller.closeConnection()

        return False


if __name__ == "__main__":

    # with keyboard.Listener(on_press=on_press) as listener:
    #     listener.join()

    while True:
        q_current = lu.getCurrentConfig()
        if q_current is not None:
            # print(q_current)
            break

    mainExperiment()
