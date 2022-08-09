import serial
from pynput import keyboard


class Controller:

    def __init__(self, portName):
        self.serial_port = serial.Serial(portName, 115200)

    def sendCommands(self, wheels_v):

        msg = "s"

        for v in wheels_v:
            msg += str(v) + '\n'

        # print(msg)

        self.serial_port.write(msg.encode())

    def openConnection():
        if (self.serial_port.isOpen() == False):
            self.serial_port.open()

    def closeConnection():
        if (self.serial_port.isOpen() == True):
            self.serial_port.close()

    def show(self, key):
        if key.char == "q":
            print('Move')
            self.sendCommands([0, 0, 4, 4, 0, 0])
        if key.char == "s":
            print('Stop')
            mainController.sendCommands([0, 0, 0, 0, 0, 0])


if __name__ == "__main__":

    portName = "COM4"
    mainController = Controller(portName)
    mainController.openConnection()

    msg_start = [4.5, 0, 0, 0, 0, 0]
    msg_stop = [0, 0, 0, 0, 0, 0]

    with keyboard.Listener(on_press=mainController.show) as listener:
        listener.join()
