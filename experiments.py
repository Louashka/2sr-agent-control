import numpy as np
import pandas as pd
import time
from datetime import datetime
from pynput import keyboard
from threading import Thread
import mainController
import LUs

expData = []
w = np.array([[2, 0, 0, 0]])
s = [[1, 0]]
flag = [False]

lu = LUs.LU()
lu.start()

portName = "COM4"
mainController = mainController.Controller(portName)
mainController.moveRobot(w, s, [True])


def unitsStabilityExpLoop():
    while True:
        # start the motion
        centers, rotation = lu.DetectArucoPose()
        timeStamp = datetime.now().strftime("%H:%M:%S")
        expData.append([timeStamp, centers[2, 0], centers[2, 1], rotation[2]
                        centers[1, 0], centers[1, 1], rotation[1]])

        mainController.moveRobot(w, s, flag)

        # time.sleep(0.05)


def on_press(key):
    global w, s, flag

    if key.char == "s":
        print('Stop')

        w = np.array([[0, 0, 0, 0]])
        s = [[0, 0]]
        flag = [True]

        columnNames = ["time stamp", "LU1 x", "LU1 y",
                       "LU1 th", "LU2 x", "LU2 y", "LU2 th"]
        df = pd.DataFrame(expData, columns=columnNames)
        print("save")
        df.to_csv('ExpData/unitsStabilityExp.csv')

        return False


if __name__ == "__main__":

    # portName = "COM4"
    # controller = mainController.Controller(portName)
    # controller.openConnection()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    Thread(target=unitsStabilityExpLoop, args=(),
           name='unitsStabilityExpLoop', daemon=True).start()

    listener.join()  # wait for abortKey

    # mainController.closeConnection()
