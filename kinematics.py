import numpy as np
import sympy as sym
import globals_


# FORWARD KINEMATICS

def fk(q_start, sigma, v, sim_time):

    dt = 0.1  # step size
    t = np.arange(dt, sim_time + dt, dt)  # span

    # Initialize a trajectory
    q_list = []
    q_list.append(q_start)

    for i in range(len(t)):
        q_current = q_list[i]  # update current configuration
        # Determine a Jacobian matrix
        J = hybridJacobian(q_start, q_current, sigma)

        # Rate of configuration change
        q_dot = np.matmul(J, v)
        # New configuration
        q_new = q_current + q_dot * dt

        # Add a new configuration to the trajectory
        q_list.append(q_new)

    return q_list


def hybridJacobian(q_start, q, sigma):

    # We divide a Jacobian matrix into 2 parts, which correspond
    # to the rigid and soft states of the 2SR robot

    # RIGID STATE

    flag_rigid = int(not (sigma[0] or sigma[1]))  # checks if both
    # segments are rigid
    J_rigid = flag_rigid * np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
                                     [np.sin(q[2]), np.cos(q[2]), 0],
                                     [0, 0, 1],
                                     [0, 0, 0],
                                     [0, 0, 0]])

    # SOFT STATE

    # Central angles of VSS arcs
    alpha = np.array(q[3:]) * globals_.L_VSS
    # Angles of 3 types of logarithmic spirals
    theta = np.divide(np.tile(np.reshape(alpha, (2, 1)), 3),
                      globals_.M) - np.pi

    # Spiral constants
    a = globals_.SPIRAL_COEF[0]
    b = np.tile(globals_.SPIRAL_COEF[1], (2, 1))
    for i in range(2):
        if q[i + 3] > 0:
            b[i] = -b[i]
            theta[i] += 2 * np.pi

    rho = a * np.exp(b * theta)  # spiral radii

    # Proportionality coefficient of the rate of VSS curvature change
    Kappa = np.divide(globals_.M, globals_.L_VSS * rho)
    # Proportionality coefficient of the rate of 2SRR orientation change
    Phi = np.divide(globals_.M, rho)

    flag_soft1 = int(not sigma[1] and sigma[0])  # checks if VSS1 is soft
    flag_soft2 = int(not sigma[0] and sigma[1])  # checks if VSS2 is soft
    flag_full_soft = int(sigma[0] and sigma[1])  # checks if both VSSs are soft

    #  Proportionality coefficients of the rate of change of 2SRR position coordinates
    Delta = np.zeros((2, 2))
    Delta[:, 0] = (flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2]) * \
        getBodyFrame(rho[1, 0], q_start[2], q[4], q_start[4], 2).ravel()
    Delta[:, 1] = (flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]) * \
        getBodyFrame(rho[0, 0], q_start[2], q[3], q_start[3], 1).ravel()

    J_soft = np.array([[-flag_soft2 * Phi[1, 1] - flag_full_soft * Phi[1, 2],
                        flag_soft1 * Phi[0, 1] + flag_full_soft * Phi[0, 2]],
                       [flag_soft1 * Kappa[0, 0] + flag_full_soft * Kappa[0, 2],
                        flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]],
                       [flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2],
                        flag_soft2 * Kappa[1, 0] + flag_full_soft * Kappa[1, 2]]])

    J_soft = np.concatenate((Delta, J_soft), axis=0)

    # complete hybrid Jacobian matrix
    J = np.concatenate((J_soft, J_rigid), axis=1)

    return J

# These functions calculate the positional coordinates of the body frame {b_0}, which is attached
# to the middle link (F point). Execution of bodyFramePosition() is time-consuming; therefore, we use
# precalculated values. But if some of the 2SR robot parameters are changed, then, bodyFramePosition()
# must be used to update the matrices in getBodyFrame()


def getBodyFrame(r, phi, k, k0, seg):

    if seg == 1:
        pos = np.array([[-0.0266 * r * np.sin(phi + 0.0266 * k - 0.04 * k0) - 0.0006 * np.sin(phi + 0.04 * k - 0.04 * k0)],
                        [0.0266 * r * np.cos(phi + 0.0266 * k - 0.04 * k0) + 0.0006 * np.cos(phi + 0.04 * k - 0.04 * k0)]])
        # pos = bodyFramePosition(-1)
    elif seg == 2:
        pos = np.array([[-0.0266 * r * np.sin(phi - 0.0266 * k + 0.04 * k0) - 0.0006 * np.sin(phi - 0.04 * k + 0.04 * k0)],
                        [0.0266 * r * np.cos(phi - 0.0266 * k + 0.04 * k0) + 0.0006 * np.cos(phi - 0.04 * k + 0.04 * k0)]])
        # pos = bodyFramePosition(1)

    return pos


def bodyFramePosition(flag):

    x = sym.Symbol('x')  # 2SRR x coordinate
    y = sym.Symbol('y')  # 2SRR y coordinate
    phi = sym.Symbol(r'\phi')  # 2SRR orientation
    k = sym.Symbol('k')  # current curvature
    k0 = sym.Symbol('k\'')  # initial curvature
    r = sym.Symbol('r')  # log spiral radius

    # Angle of the 1st log spiral
    th = k * globals_.L_VSS / globals_.M[0]

    # Coordinates of the VSS G point w.r.t. the log spiral cenre {c}
    c_G = np.array([[-flag * r * sym.cos(th)],
                    [r * sym.sin(th)],
                    [1]])

    # Coordinates of the F point (middle point of the middle link) w.r.t. the G
    G_F = np.array([[-flag * globals_.L_LINK / 2],
                    [0],
                    [1]])

    # Linear transformation from the G point to the {c} frame
    cGth = - flag * k * globals_.L_VSS
    c_T_G = np.array([[sym.cos(cGth), -sym.sin(cGth), c_G[0].item()],
                      [sym.sin(cGth), sym.cos(cGth), c_G[1].item()],
                      [0, 0, 1]])

    # Coordinates of the F point w.r.t. the {c} frame
    c_F = sym.simplify(np.matmul(c_T_G, G_F))

    # Linear transformation from the {c} frame to the VSS end frame {b_j}
    b_T_c = np.array([[1, 0, globals_.SPIRAL_CENTRE[0]],
                      [0, 1, globals_.SPIRAL_CENTRE[1]],
                      [0, 0, 1]])

    # Coordinates of the F point w.r.t. the {b_j} frame
    b_F = sym.simplify(np.matmul(b_T_c, c_F))

    # Linear transformation from the {b_j} frame to the body frame {b_0}
    o_T_b0 = np.array([[sym.cos(phi), -sym.sin(phi), x],
                       [sym.sin(phi), sym.cos(phi), y],
                       [0, 0, 1]])

    # Linear transformation from the VSS end frame {b_j} to the body frame {b_0}
    b0Bth = flag * k0 * globals_.L_VSS
    b0_T_b = np.array([[sym.cos(b0Bth), -sym.sin(b0Bth), flag * globals_.L_LINK / 2 + sym.sin(b0Bth) / k0],
                       [sym.sin(b0Bth), sym.cos(b0Bth),
                        (1 - sym.cos(b0Bth)) / k0],
                       [0, 0, 1]])

    # Linear transformation from the body frame {b_0} to the global frame {o}
    o_T_b = sym.simplify(np.matmul(o_T_b0, b0_T_b))

    # Coordinates of the F point w.r.t. the {o} frame
    o_F = sym.simplify(np.matmul(o_T_b, b_F))

    # Take a partial derivative of the F positional coordinates w.r.t. VSS curvature k
    diff = sym.Matrix(sym.simplify(sym.diff(o_F, k))[:2])

    return diff
