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
q_target = []


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

    dist = np.linalg.norm(q_current - q_target)

    while dist > 10**(-5):
        config = controller.motionPlanner(q_current, q_target)
        w = controller.wheelDrive(config[0], config[1], config[2])

        controller.moveRobot(w, config[2], config[3])

        q_current = lu.getCurrentConfig()

        error = np.linalg.norm(config[0] - q_current)
        dist = np.linalg.norm(q_current - q_target)

        expData.append([error, dist])

    w = np.array([[0, 0, 0, 0]])
    s = [0, 0]
    flag = True
    controller.moveRobot(w, s, flag)

    columnNames = ["error", "dist"]
    df = pd.DataFrame(expData, columns=columnNames)
    print("save")
    df.to_csv('ExpData/mainExperiment1.csv', index=False)


def on_press(key):
    global w, s, flag

    if key.char == "s":
        print('Stop and save')

        w = np.array([[0, 0, 0, 0]])
        s = [[0, 0]]
        flag = [True]
        controller.moveRobot(w, s, flag)

        # columnNames = ["time stamp", "LU1 x", "LU1 y",
        #                "LU1 th", "LU2 x", "LU2 y", "LU2 th"]
        columnNames = ["error", "dist"]
        df = pd.DataFrame(expData, columns=columnNames)
        print("save")
        df.to_csv('ExpData/mainExperiment1.csv', index=False)

        return False


if __name__ == "__main__":

    # portName = "COM4"
    # controller = mainController.Controller(portName)
    # controller.openConnection()

    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()

    mainExperiment()

    # Thread(target=mainExperiment, args=(),
    #        name='mainExperiment', daemon=True).start()

    # listener.join()  # wait for abortKey

    # mainController.closeConnection()
