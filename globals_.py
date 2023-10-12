import math
# Bridge parameters

L_VSS = 40 * 10**(-3)  # VSS length
L_LINK = 23 * 10**(-3)  # plastic link length
L_LINK_END = 30 * 10**(-3)
D_BRIDGE = 7 * 10**(-3)  # bridge diameter
L_VSB = 2 * L_VSS + L_LINK  # VSB length

# Moving bloc parameters

BLOCK_SIDE = 42 * 10**(-3)  # block side length
WHEEL_R = 10 * 10**(-3)  # wheel radius
WHEEL_TH = 15 * 10**(-3)  # wheel thickness

BETA = [math.pi / 2, math.pi, -math.pi / 2, math.pi]

H1 = L_LINK_END + BLOCK_SIDE / 2
H2 = BLOCK_SIDE - D_BRIDGE / 2 + WHEEL_TH / 2
H3 = L_LINK_END - WHEEL_TH / 2

# Wheels coordinates w.r.t. to VSB end frames {b_j}
bj_Q_w = [[-H3, -H1, H3, H1],
          [0, -H2, 0, -H2]]


# Constants of logarithmic spirals

SPIRAL_COEF = [[2.3250 * L_VSS, 3.3041 * L_VSS,
                2.4471 * L_VSS], [0.3165, 0.083, 0.2229]]

SPIRAL_CENTRE = [-0.1223 * L_VSS, 0.1782 * L_VSS]

M = [3 / 2, 1, 3 / 4]
