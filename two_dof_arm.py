import numpy as np


def getForwardModel(q0, q1):
    """
    Takes joint states(angles in radians) and returns end effector positions
    """

    assert all(isinstance(i, (int, float))for i in (q0, q1)), "Angles have to be floats or integers"
    assert all(-math.pi <= i <= math.pi for i in (q0, q1)), "Angle should be radians between -Pi and +Pi"

    



