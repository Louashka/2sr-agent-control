import mainController
from pynput import keyboard
import numpy as np

portName = "/dev/tty.usbserial-0001"
controller = mainController.Controller(portName)

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
s_current = [0, 0]
s = [0, 0]
agent_id = 2


def manualControl(v, s, agent_id):
    global s_current
    # print(s_current)
    # print(s)

    if agent_id == 3:
        q_current1 = [0, 0, 0, 20, 25]
        q_current2 = [0, 0, 0, 20, 25]

        w1 = controller.wheelDrive(q_current1, v, s) * 0.5
        w2 = controller.wheelDrive(q_current2, v, s) * 0.5

        controller.moveRobotM(w1, s, 1)
        controller.moveRobotM(w2, s, 2)

    else:
        q_current = [0, 0, 0, 0, 0]

        w = controller.wheelDrive(q_current, v, s)

        flag_cool = False
        if s_current[0] == 1 and s[0] == 0 or s_current[1] == 1 and s[1] == 0:
            flag_cool = True

        flag = False
        if s[0] != s_current[0] or s[1] != s_current[1]:
            flag = True
            s_current = s

        controller.moveRobotM(w, s, agent_id)


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
            v[0] = -lu_speed
            v[1] = -lu_speed

    if key in BOTH_LU_BACKWARD:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_LU_BACKWARD):
            print("LUs backward")
            flag = False
            v[0] = lu_speed
            v[1] = lu_speed

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

    manualControl(v, s, agent_id)


def on_release(key):
    global current_keys
    try:
        current_keys.remove(key)
        controller.moveRobotM(np.array([0, 0, 0, 0]), s_current, agent_id)
    except KeyError:
        pass


if __name__ == "__main__":
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
