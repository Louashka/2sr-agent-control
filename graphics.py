import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np
import globals_


LINK_DIAG = ((globals_.L_LINK / 2)**2 + (globals_.D_BRIDGE / 2)**2)**(1 / 2)
fig, ax = plt.subplots()
q_list = []
q_target = []
s_array = []

x_range = 0
y_range = 0

link = Rectangle((0, 0), 0, 0, fc='y')
arc1, = ax.plot([], [], lw=2, color="blue")
arc2, = ax.plot([], [], lw=2, color="blue")
centre, = ax.plot([], [], lw=2, marker=".", color="black")

stiffness_text = ax.text(0, 0, '', fontsize=12)

target_link = Rectangle((0, 0), 0, 0, fc='y', alpha=0.5)
target_arc1, = ax.plot([], [], lw=1, color="blue", alpha=0.5)
target_arc2, = ax.plot([], [], lw=1, color="blue", alpha=0.5)


def init():
    global ax, x_range, y_range

    x_range, y_range = defineRange()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    ax.add_patch(link)

    if q_target:
        ax.add_patch(target_link)


def defineRange():
    margin = 0.1

    q_array = np.array(q_list)
    x_min, y_min = q_array[:, :2].min(axis=0)
    x_max, y_max = q_array[:, :2].max(axis=0)

    ax_range = max(x_max - x_min, y_max - y_min) + margin

    x_range = (x_min - margin, x_min + ax_range)
    y_range = (y_min - margin, y_min + ax_range)

    return x_range, y_range


def genArc(q, seg):
    s = np.linspace(0, globals_.L_VSS, 50)

    flag = -1 if seg == 1 else 1

    gamma_array = q[2] + flag * q[2 + seg] * s

    x_0 = q[0] + flag * np.cos(q[2]) * globals_.L_LINK / 2
    y_0 = q[1] + flag * np.sin(q[2]) * globals_.L_LINK / 2

    if q[2 + seg] == 0:
        x = x_0 + [0, flag * globals_.L_VSS * np.cos(q[2])]
        y = y_0 + [0, flag * globals_.L_VSS * np.sin(q[2])]
    else:
        x = x_0 + np.sin(gamma_array) / \
            q[2 + seg] - np.sin(q[2]) / q[2 + seg]
        y = y_0 - np.cos(gamma_array) / \
            q[2 + seg] + np.cos(q[2]) / q[2 + seg]

    return [x, y]


def update(i):
    global link, target_link, arc1, arc2, centre, stiffness_text, target_arc1, target_arc2
    q = q_list[i]

    x = q[0]
    y = q[1]
    phi = q[2]

    x0 = x - globals_.L_LINK / 2
    y0 = y - globals_.D_BRIDGE / 2

    link.set_width(globals_.L_LINK)
    link.set_height(globals_.D_BRIDGE)
    link.set_xy([x0, y0])

    transform = mpl.transforms.Affine2D().rotate_around(
        x, y, phi) + ax.transData
    link.set_transform(transform)

    seg1 = genArc(q, 1)
    seg2 = genArc(q, 2)

    arc1.set_data(seg1[0], seg1[1])
    arc2.set_data(seg2[0], seg2[1])

    if s_array[i][0] == 0:
        arc1.set_color("red")
    else:
        arc1.set_color("blue")

    if s_array[i][1] == 0:
        arc2.set_color("red")
    else:
        arc2.set_color("blue")

    centre.set_data(x, y)

    stiffness_text.set_text(
        "s1: " + str(s_array[i][0]) + ", s2: " + str(s_array[i][1]))
    stiffness_text.set_position(
        (x_range[1] - (x_range[1] - x_range[0]) / 3.5, y_range[1] - (y_range[1] - y_range[0]) / 15))

    if q_target:

        x_t = q_target[0] - globals_.L_LINK / 2
        y_t = q_target[1] - globals_.D_BRIDGE / 2
        phi_t = q_target[2]

        target_link.set_width(globals_.L_LINK)
        target_link.set_height(globals_.D_BRIDGE)
        target_link.set_xy([x_t, y_t])

        target_transform = mpl.transforms.Affine2D().rotate_around(
            q_target[0], q_target[1], phi_t) + ax.transData
        target_link.set_transform(target_transform)

        target_seg1 = genArc(q_target, 1)
        target_seg2 = genArc(q_target, 2)

        target_arc1.set_data(target_seg1[0], target_seg1[1])
        target_arc2.set_data(target_seg2[0], target_seg2[1])

        return link, arc1, arc2, centre, stiffness_text, target_link, target_arc1, target_arc2,

    return link, arc1, arc2, centre, stiffness_text,


def plotMotion(q, s, frames, q_t=[]):
    global q_list, s_array, q_target
    q_list = q
    s_array = s
    q_target = q_t

    anim = FuncAnimation(fig, update, frames,
                         init_func=init, interval=1, repeat=True)

    # Save animation
    mywriter = FFMpegWriter(fps=30)
    anim.save('experiment2.mp4', writer=mywriter)

    plt.show()
