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


def getIK(x,y, q0_current= 0, q1_current=0):
    """
    Inverse kinematics using Jacobian Inverse

    :param x: Position x
    :param y: Position y
    :return: Joint angles
    """
    assert all(isinstance(i, (int, float)) for i in (x, y)), "Positions have to be floats or integers"
    assert all(isinstance(i, (int, float)) for i in (q0_current, q1_current)), "Angles have to be floats or integers"

    inverse_jacobian = np.linalg.pinv(getJacobian(q0_current,q1_current))
    assert inverse_jacobian.shape == (2,3)

    current_position = np.array([x,y,0]).reshape(-1,1)

    del_q = inverse_jacobian@current_position
    assert del_q.shape == (2,1)

    return del_q


def get_samples_from_trajectory(steps=100):
    """
    Create x_desired, y_desired from the trajectory equation
    Pack into numpy array and return
    :param steps: Number of sample points needed from trajectory
    :return: Desired trajectory : numpy array of shape (2,n) : First row refers to x, Second row refers to y
    """

    assert isinstance(steps, int), "Steps has to be integer"

    desired_traj = np.zeros(shape=(2, steps), dtype=float)
    theta_values = np.linspace(start=-np.pi, stop=np.pi, num=steps)

    c1, c2 = 0.19, 0.02
    desired_traj[0, :] = (c1 + c2*np.cos(4*theta_values))*np.cos(theta_values)
    desired_traj[1, :] = (c1 + c2*np.cos(4*theta_values))*np.sin(theta_values)

    return desired_traj


def create_trajectory(steps=100, kp_1=1.0, kp_2=1.0, kd_1=1.0, kd_2=1.0, episodes=1):
    """
    Given steps, Proportion Control(kp_1, kp_2) and Differential Control parameters (kd_1, kd_2) return observed trajectory (x_obs, y_obs)
    :param episodes:
    :param steps:
    :param kp_1:
    :param kp_2:
    :param kd_1:
    :param kd_2:
    :return:
    """
    assert isinstance(steps, int), "steps has to be integer"
    assert all(isinstance(i, (float, int)) for i in (kp_1, kp_2, kd_1, kd_2)),"PD controller gains should be integer, float"

    import gym
    import pybulletgym.envs
    env = gym.make("ReacherPyBulletEnv-v0")
    q0_curr = 0  # Initial position
    q1_curr = 0

    for curr_episode in range(episodes):  # For multiple episodes, Default: episodes= 1
        env.unwrapped.robot.central_joint.reset_position(q0_curr, 0)
        env.unwrapped.robot.elbow_joint.reset_position(q1_curr, 0)
        for robo_step in range(steps):
            env.render()  # WHY HERE?
            q0_obs, q0_dot = env.unwrapped.robot.central_joint.current_position()  # Current Observation from Sensor
            q1_obs, q1_dot = env.unwrapped.robot.elbow_joint.current_position()

            print("\nJoint 1", q0_obs)
            print("Joint 2", q1_obs)

            #action = env.action_space.sample() #[0.5, 0.7] Sample action. Torque for q0, q1
            action = get_torque(q0_obs, q1_obs, kp_1, kp_2, kd_1, kd_2)
            _, _, done, _ = env.step(action)
            if done:
                print(f"Episode finished after {robo_step} steps")
                break
    env.close()


def get_torque(q0_obs, q1_obs, *args):
    """
    Given Current observation and PD controller terms, return Torque (action for env action space)
    :param q0_obs:
    :param q1_obs:
    :param args:
    :return:
    """

    kp_1, kp_2, kd_1, kd_2 = args
    assert all(isinstance(i, (float, int)) for i in (kp_1, kp_2, kd_1, kd_2)), "PD controller gains should be integer, float"

    tau_1 = 0.0
    tau_2 = 0.0
    action = np.array([tau_1, tau_2], dtype=float)

    return action

