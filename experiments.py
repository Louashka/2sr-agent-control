import pandas as pd
import time
from datetime import datetime
from pynput import keyboard
from threading import Thread
import mainController

expData = []


def unitsStabilityExpLoop():
    while True:
        # start the motion
        timeStamp = datetime.now().strftime("%H:%M:%S")
        expData.append([timeStamp, 0, 0])
        time.sleep(0.05)


def on_press(key):

    if key.char == "s":
        print('Stop')

        columnNames = ["time stamp", "LU1 position", "LU2 position"]
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
