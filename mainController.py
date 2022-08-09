import serial
import numpy as np
import kinematics
import graphics
import random as rnd
from pynput import keyboard


class Controller:

    def __init__(self, portName):
        # self.serial_port = serial.Serial(portName, 115200)
        self.serial_port = None

    def motionPlanner(self, q_0, q_target):
        dt = 0.1  # step size
        # A set of possible stiffness configurations
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Initialize a sequence of VSB stiffness values
        s_list = []
        # 2SRR always starts from the rigid state
        # s_list.append(s[0])
        # Initialize the number of stiffness transitions
        switch_counter = 0

        # Initialize a trajectory
        q_list = []
        q = q_0  # current configuration
        # q_list.append(q)

        # Initialize a sequence of velocity commands
        v_list = []

        # A set of possible configurations and velocities
        q_ = [None] * len(s)
        v_ = [None] * len(s)

        q_t = np.array(q_target)
        # Euclidean distance between current and target configurations (error)
        dist = np.linalg.norm(q - q_t)

        t = 0.1  # current time
        # feedback gain
        velocity_coeff = np.ones((5,), dtype=int)
        # Index of the current stiffness configuration
        current_i = None

        while dist > 0:

            flag = False  # indicates whether VSB stiffness has changed

            # INVERSE KINEMATICS

            q_tilda = velocity_coeff * (q_t - q) * t
            for i in range(len(s)):
                # Jacobian matrix
                J = kinematics.hybridJacobian(q_0, q, s[i])
                # velocity input commands
                v_[i] = np.matmul(np.linalg.pinv(J), q_tilda)
                q_dot = np.matmul(J, v_[i])
                q_[i] = q + (1 - np.exp(-1 * t)) * q_dot * dt

            # Determine the stiffness configuration that promotes
            # faster approach to the target
            dist_ = np.linalg.norm(q_ - q_t, axis=1)
            min_i = np.argmin(dist_)

            # The extent of the configuration change
            delta_q_ = np.linalg.norm(q - np.array(q_), axis=1)

            # Stiffness transition is committed only if the previous
            # stiffness configuration does not promote further motion
            if min_i != current_i and current_i is not None:
                if delta_q_[current_i] > 10**(-17):
                    min_i = current_i
                else:
                    flag = True

            current_i = min_i  # update current stiffness
            q = q_[current_i]  # update current configuration
            dist = np.linalg.norm(q - q_t)  # update error

            if (delta_q_[current_i] > 10 ** (-5)):
                q_list.append(q)
                v_list.append(v_[current_i])
                s_list.append(s[current_i])

                if flag:
                    switch_counter += 1

            t += dt  # increment time

        return q_list, v_list, s_list, switch_counter

    def wheelDrive(self, v, s):
        w = [0, 0, 0, 0]
        return w

    def sendCommands(self, wheels_v_list, s_list):
        return

    def sendData(self, q_0, commands):

        msg = "s"

        for command in commands:
            msg += str(command) + '\n'

        # print(msg)

        self.serial_port.write(msg.encode())

    def openConnection():
        if self.serial_port.isOpen() is False:
            self.serial_port.open()

    def closeConnection():
        if self.serial_port.isOpen() is True:
            self.serial_port.close()

    def show(self, key):
        if key.char == "q":
            print('Move')
            self.sendData([0, 0, 4, 4, 0, 0])
        if key.char == "s":
            print('Stop')
            mainController.sendData([0, 0, 0, 0, 0, 0])


if __name__ == "__main__":

    portName = "COM4"
    mainController = Controller(portName)
    mainController.openConnection()

    config = mainController.motionPlanner(q_start, q_target)
    wheels_v_list = mainController.wheelDrive(config[1], config[2])
    mainController.sendCommands(wheels_v_list, config[2])

    mainController.closeConnection()

    # msg_start = [4.5, 0, 0, 0, 0, 0]
    # msg_stop = [0, 0, 0, 0, 0, 0]

    # with keyboard.Listener(on_press=mainController.show) as listener:
    #     listener.join()
