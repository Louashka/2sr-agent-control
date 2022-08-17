import pandas as pd
import time
from datetime import datetime
from pynput import keyboard
from threading import Thread
import pyrealsense2 as rs
import cv2
# import mainController
import LUs

expData = []

lu = LUs.LU()
lu.start()


def unitsStabilityExpLoop():
    while True:
        # start the motion
        centers, rotationVec = lu.DetectArucoPose()

        timeStamp = datetime.now().strftime("%H:%M:%S")
        expData.append([timeStamp, centers[2, 0], centers[2, 1],
                        centers[1, 0], centers[1, 1]])
        # time.sleep(0.05)


def on_press(key):
    if key.char == "s":
        print('Stop')
        columnNames = ["time stamp", "LU1 x", "LU1 y", "LU2 x", "LU2 y"]
        df = pd.DataFrame(expData, columns=columnNames)
        print("save")
        df.to_csv('unitsStabilityExp.csv')
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

lu.stop()