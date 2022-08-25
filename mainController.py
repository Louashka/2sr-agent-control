import serial
import time
import numpy as np
import kinematics
import graphics
import random as rnd
from pynput import keyboard
import globals_


class Controller:

    def __init__(self, portName):
        self.serial_port = serial.Serial(portName, 115200)
        self.vss_on = False
        self.counter = 0
        # self.serial_port = None

    def motionPlanner(self, q, q_target, s_current):
        dt = 0.1  # step size
        # A set of possible stiffness configurations
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Initialize a sequence of VSB stiffness values

        # A set of possible configurations and velocities
        q_ = [None] * len(s)
        v_ = [None] * len(s)

        q_t = np.array(q_target)
        # Euclidean distance between current and target configurations (error)
        dist = np.linalg.norm(q - q_t)

        t = 0.1  # current time
        # feedback gain
        velocity_coeff = 1 * np.ones((5,), dtype=int)
        # Index of the current stiffness configuration
        current_i = None

        flag = False  # indicates whether VSB stiffness has changed

        # INVERSE KINEMATICS

        q_tilda = velocity_coeff * (q_t - q) * t
        for i in range(len(s)):
            # Jacobian matrix
            J = kinematics.hybridJacobian(q, q, s[i])
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

        current_i = min_i  # update current stiffness
        q_new = q_[current_i]  # update current configuration
        v_new = v_[current_i]
        s_new = s[current_i]
        dist = np.linalg.norm(q_new - q_t)  # update error

        if s_new != s_current:
            flag = True

        if (delta_q_[current_i] < 10 ** (-5)):
            q_new, v_new, s_new, flag = motionPlanner(self, q_new, q_target)

        return q_new, v_new, s_new, flag

    def wheelDrive(self, q, v, s):

        flag_soft = int(s[0] or s[1])
        flag_rigid = int(not (s[0] or s[1]))

        k = [q[3], q[3], q[4], q[4]]

        V_ = np.zeros((4, 5))
        for i in range(4):
            b0_q_w = self.getWheelPosition(i, k[i], q[2])
            tau = b0_q_w[0] * np.sin(b0_q_w[2]) - \
                b0_q_w[1] * np.cos(b0_q_w[2])
            V_[i, :] = [flag_soft * int(i == 0), -flag_soft * int(
                i == 2), flag_rigid * np.cos(b0_q_w[2]), flag_rigid * np.sin(b0_q_w[2]), flag_rigid * tau]

        V = 1 / globals_.WHEEL_R * V_
        w = np.matmul(V, v)

        return w.round(3)

    def getWheelPosition(self, i, k, phi):

        flag = 1 if i < 2 else -1
        alpha = k * globals_.L_VSS

        if k == 0:
            b0_x_bj = globals_.L_LINK / 2 + globals_.L_VSS
            b0_y_bj = 0
        else:
            b0_x_bj = globals_.L_LINK / 2 + np.sin(alpha) / k
            b0_y_bj = (1 - np.cos(alpha)) / k

        b0_T_bj = np.array([[np.cos(alpha), flag * np.sin(alpha), -flag * b0_x_bj],
                            [-flag * np.sin(alpha), np.cos(alpha), b0_y_bj],
                            [0, 0, 1]])

        b0_q_w = np.matmul(b0_T_bj, np.append(
            globals_.bj_Q_w, [[1, 1, 1, 1]], axis=0)[:, i])

        b0_q_w = np.append(b0_q_w[:-1], phi - flag * alpha + globals_.BETA[i])

        return b0_q_w

    def moveRobot(self, w, s, flag):

        w *= 7
        commands = w.tolist() + s
        commands_ = w.tolist() + [0, 0]

        if (flag):
            self.sendData([0, 0, 0, 0] + s)
            self.sendData([0, 0, 0, 0, 0, 0])
            print("Phase transition: ", s)
            self.counter = 0
            if all(s) == 1:
                time.sleep(45)
            else:
                time.sleep(90)

        if self.counter < 75:
            if self.vss_on == True:
                self.sendData(commands)
                print("Controller commands (ON): ", commands)
            else:
                self.sendData(commands_)
                print("Controller commands (OFF): ", commands_)
            self.counter += 1
        else:
            self.sendData(commands)
            print("Controller commands (timer 0): ", commands)
            self.counter = 0
            self.vss_on = not self.vss_on

        time.sleep(0.01)

    def sendData(self, commands):

        msg = "s"

        for command in commands:
            msg += str(command) + '\n'

        # print(msg.encode())

        self.serial_port.write(msg.encode())

    def openConnection(self):
        if self.serial_port.isOpen() is False:
            self.serial_port.open()

    def closeConnection(self):
        if self.serial_port.isOpen() is True:
            self.serial_port.close()

    def show(self, key):
        if key.char == "q":
            print('Move')
            self.sendData([0, 0, 4, 4, 0, 0])
        if key.char == "s":
            print('Stop')
            mainController.sendData([0, 0, 0, 0, 0, 0])


# if __name__ == "__main__":

#     portName = "COM4"
#     mainController = Controller(portName)
#     # mainController.openConnection()

#     # SIMULATION PARAMETERS

#     sim_time = 15  # simulation time
#     dt = 0.1  # step size
#     t = np.arange(dt, sim_time + dt, dt)  # span
#     frames = len(t)  # number of frames

#     # Initial configuration
#     q_start = [0, 0, 0, 0, 0]

#     # FORWARD KINEMATICS

#     # # Stiffness of the VS segments
#     # sigma = [rnd.randint(0, 1), rnd.randint(0, 1)]
#     # print("Stiffness: ", sigma)
#     # # Input velocity commands
#     # v = [rnd.uniform(-0.007, 0.007), rnd.uniform(-0.007, 0.007),
#     #      rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03), rnd.uniform(-0.1, 0.1)]
#     # print("Velocity: ", v)

#     # sigma = [1, 1]
#     # v = [0.00, 0.003, 0.0, 0.00, 0]

#     sigma = [1, 1]
#     v = [0.0055, 0.0035, 0.0, 0.00, 0]

#     # sigma = [0, 1]
#     # v = [-0.0021322214883674933, 0.0038127879372345154, -
#     #      0.00134946847859305, -0.0010331590290957905, -0.026437336778516826]

#     # Generate a trajectory by an FK model
#     q_list = kinematics.fk(q_start, sigma, v, sim_time)

#     # MOTION PLANNER (STIFFNESS PLANNER + INVERSE KINEMATICS)

#     # We take the last configuration of an FK trajectory
#     # as a target configuration
#     q_target = q_list[-1].tolist()

#     config = mainController.motionPlanner(q_start, q_target)
#     w_list = mainController.wheelDrive(config[0], config[1], config[2])
#     print(config[2])

#     frames = len(config[0])

#     # Animation of the 2SRR motion towards the target
#     graphics.plotMotion(config[0], config[2], frames, q_t=q_target)

#     # mainController.moveRobot(w_list, config[2], config[3])

#     # mainController.closeConnection()

#     # msg_start = [4.5, 0, 0, 0, 0, 0]
#     # msg_stop = [0, 0, 0, 0, 0, 0]

#     # with keyboard.Listener(on_press=mainController.show) as listener:
#     #     listener.join()
