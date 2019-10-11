import numpy as np
import math


def getForwardModel(q0, q1, l0 = 0.1, l1 = 0.11 ):
    """
    Takes joint states(angles in radians) and returns end effector pose (3,1) numpy array
    """

    assert all(isinstance(i, (int, float))for i in (q0, q1)), "Angles have to be floats or integers"
    assert all(-math.pi <= i <= math.pi for i in (q0, q1)), "Angle should be radians between -Pi and +Pi"

    pos_x = l1 * np.cos(q0+q1) + l0*np.cos(q0)
    pos_y = l1 * np.sin(q0+q1) + l0*np.sin(q0)
    theta_z = q0+q1

    state = np.array([pos_x, pos_y, theta_z]).reshape(-1, 1)
    assert state.shape == (3,1)

    return state


def getJacobian(q0, q1, l0=0.1, l1=0.11):
    """
    Takes joint states(angles in radians) and returns Jacobian (3,2) np array
    """

    assert all(isinstance(i, (int, float)) for i in (q0, q1)), "Angles have to be floats or integers"
    assert all(-math.pi <= i <= math.pi for i in (q0, q1)), "Angle should be radians between -Pi and +Pi"

    jacobian = np.ones(shape=(3,2), dtype=float)

    jacobian[0][0] = -(l1 * np.sin(q0+q1) + l0*np.sin(q0))
    jacobian[0][1] = -l1*np.sin(q0+q1)
    jacobian[1][0] = l1 * np.cos(q0+q1) + l0*np.cos(q0)
    jacobian[1][1] = l1*np.cos(q0+q1)

    return jacobian


