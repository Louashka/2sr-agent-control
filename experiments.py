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
# q_target = [0.28, 0.24, -1, 23, 19]
# q_target = [0.28, 0.14, -0.4, -29, 27]
# q_target1 = [0.24, 0.21, 0.5, 28, 3]
q_target = [0.253,   0.284,  -0.5, 3, 32]
s_current = [0, 0]
s = [0, 0]

portName = "COM4"
controller = mainController.Controller(portName)
# controller.moveRobot(w, s, [True])

ROTATION_LEFT = {keyboard.KeyCode.from_char('r'), keyboard.Key.left}
ROTATION_RIGHT = {keyboard.KeyCode.from_char('r'), keyboard.Key.right}

LEFT_LU_FORWARD = {keyboard.KeyCode.from_char('1'), keyboard.Key.up}
LEFT_LU_BACKWARD = {keyboard.KeyCode.from_char('1'), keyboard.Key.down}

RIGHT_LU_FORWARD = {keyboard.KeyCode.from_char('2'), keyboard.Key.up}
RIGHT_LU_BACKWARD = {keyboard.KeyCode.from_char('2'), keyboard.Key.down}
BOTH_LU_FORWARD = {keyboard.KeyCode.from_char('3'), keyboard.Key.up}
BOTH_LU_BACKWARD = {keyboard.KeyCode.from_char('3'), keyboard.Key.down}

S1_SOFT = {keyboard.Key.left, keyboard.KeyCode.from_char('s')}
S1_RIGID = {keyboard.Key.left, keyboard.KeyCode.from_char('x')}
S2_SOFT = {keyboard.Key.right, keyboard.KeyCode.from_char('s')}
S2_RIGID = {keyboard.Key.right, keyboard.KeyCode.from_char('x')}
BOTH_SOFT = {keyboard.Key.up, keyboard.KeyCode.from_char('s')}
BOTH_RIGID = {keyboard.Key.down, keyboard.KeyCode.from_char('x')}

# The currently active modifiers
current_keys = set()


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
    flag_cool = False

    while dist > 10**(-2):

        q_current = lu.getCurrentConfig()
        if q_current is None:
            continue
        print("Current config: ", q_current)
        config = controller.motionPlanner(q_current, q_target, s_current)
        w = controller.wheelDrive(q_current, config[1], config[2])

        if s_current[0] == 1 and config[2][0] == 0 or s_current[1] == 1 and config[2][1] == 0:
            flag_cool = True
        else:
            flag_cool = False
        # print("Flag: ", flag_cool)

        controller.moveRobot(w, config[2], config[3], flag_cool)

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
    flag = False
    controller.moveRobot(w, s, flag, False)

    columnNames = ["x_model", "y_model", "angle_model", "k1_model", "k2_model", "v1", "v2", "u0", "v0", "r0", "x", "y", "angle", "k1", "k2",
                   "s1", "s2", "time"]
    df = pd.DataFrame(expData, columns=columnNames)
    print("save")
    df.to_csv('ExpData/mainExperiment7.csv', index=False)

    controller.closeConnection()


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

            flag_cool = False
            if s_current[0] == 1 and s[0] == 0 or s_current[1] == 1 and s[1] == 0:
                flag_cool = True

            flag = False
            if s[0] != s_current[0] or s[1] != s_current[1]:
                flag = True
                s_current = s

            controller.moveRobotM(w, s)

            break


def on_press(key):
    global current_keys, s

    flag = True

    v = [0] * 5

    omni_speed = 0.1
    rotation_speed = 0.7
    lu_speed = 0.1

    if key in ROTATION_LEFT:
        current_keys.add(key)
        if all(k in current_keys for k in ROTATION_LEFT):
            print('Turn left')
            flag = False
            v[4] = rotation_speed

    if key in ROTATION_RIGHT:
        current_keys.add(key)
        if all(k in current_keys for k in ROTATION_RIGHT):
            print('Turn right')
            flag = False
            v[4] = -rotation_speed

    if key in LEFT_LU_FORWARD:
        current_keys.add(key)
        if all(k in current_keys for k in LEFT_LU_FORWARD):
            print("LU1 forward")
            flag = False
            v[0] = lu_speed

    if key in LEFT_LU_BACKWARD:
        current_keys.add(key)
        if all(k in current_keys for k in LEFT_LU_BACKWARD):
            print("LU1 backward")
            flag = False
            v[0] = -lu_speed

    if key in RIGHT_LU_FORWARD:
        current_keys.add(key)
        if all(k in current_keys for k in RIGHT_LU_FORWARD):
            print("LU2 forward")
            flag = False
            v[1] = lu_speed

    if key in RIGHT_LU_BACKWARD:
        current_keys.add(key)
        if all(k in current_keys for k in RIGHT_LU_BACKWARD):
            print("LU2 backward")
            flag = False
            v[1] = -lu_speed

    if key in BOTH_LU_FORWARD:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_LU_FORWARD):
            print("LUs forward")
            flag = False
            v[0] = lu_speed
            v[1] = lu_speed

    if key in BOTH_LU_BACKWARD:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_LU_BACKWARD):
            print("LUs backward")
            flag = False
            v[0] = -lu_speed
            v[1] = -lu_speed

    if key in S1_SOFT:
        current_keys.add(key)
        if all(k in current_keys for k in S1_SOFT):
            print("Segment 1 soft")
            flag = False
            s[0] = 1

    if key in S1_RIGID:
        current_keys.add(key)
        if all(k in current_keys for k in S1_RIGID):
            print("Segment 1 rigid")
            flag = False
            s[0] = 0

    if key in S2_SOFT:
        current_keys.add(key)
        if all(k in current_keys for k in S2_SOFT):
            print("Segment 2 soft")
            flag = False
            s[1] = 1

    if key in S2_RIGID:
        current_keys.add(key)
        if all(k in current_keys for k in S2_RIGID):
            print("Segment 2 rigid")
            flag = False
            s[1] = 0

    if key in BOTH_SOFT:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_SOFT):
            print("Both segments soft")
            flag = False
            s = [1, 1]

    if key in BOTH_RIGID:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_RIGID):
            print("Both segments rigid")
            flag = False
            s = [0, 0]

    if flag:

        if key == keyboard.Key.up:
            print("forward")
            v[3] = omni_speed

        if key == keyboard.Key.down:
            print("backward")
            v[3] = -omni_speed

        if key == keyboard.Key.left:
            print("left")
            v[2] = -omni_speed

        if key == keyboard.Key.right:
            print("right")
            v[2] = omni_speed

    manualControl(v, s)


def on_release(key):
    global current_keys
    try:
        current_keys.remove(key)
        controller.moveRobotM(np.array([0, 0, 0, 0]), s_current)
    except KeyError:
        pass


if __name__ == "__main__":

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # while True:
    #     q_current = lu.getCurrentConfig()
    #     if q_current is not None:
    #         # print("Initial config: ", q_current)
    #         # print(controller.getWheelPosition(1, q_current[3]))
    #         break

    # mainExperiment()
