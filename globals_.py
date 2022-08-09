L_VSS = 40 * 10**(-3)  # VSS length
L_LINK = 30 * 10**(-3)  # plastic link length
D_BRIDGE = 7 * 10**(-3)  # bridge diameter
L_VSB = 2 * L_VSS + L_LINK  # VSB length

# Constants of logarithmic spirals

SPIRAL_COEF = [[2.3250 * L_VSS, 3.3041 * L_VSS,
                2.4471 * L_VSS], [0.3165, 0.083, 0.2229]]

SPIRAL_CENTRE = [-0.1223 * L_VSS, 0.1782 * L_VSS]

M = [3 / 2, 1, 3 / 4]
