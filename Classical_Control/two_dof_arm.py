import numpy as np
import matplotlib.pyplot as plt
import gym
import pybulletgym.envs


def getForwardModel(q0, q1,
                    l0=0.1, l1=0.11):
    """
    Takes joint states(angles in radians) and returns end effector pose (3,1) numpy array
    :param q0:
    :param q1:
    :param l0:
    :param l1:
    :return:
    """
    assert all(isinstance(i, (int, float))for i in (q0, q1)), "Angles have to be floats or integers"

    pos_x = l1 * np.cos(q0+q1) + l0*np.cos(q0)
    pos_y = l1 * np.sin(q0+q1) + l0*np.sin(q0)
    theta_z = q0+q1

    state = np.array([pos_x, pos_y, theta_z])
    assert state.shape == (3,)

    return state


def getJacobian(q0, q1,
                l0=0.1, l1=0.11):
    """
    Takes joint states(angles in radians) and returns Jacobian (3,2) np array
    :param q0:
    :param q1:
    :param l0:
    :param l1:
    :return:
    """
    assert all(isinstance(i, (int, float)) for i in (q0, q1)), "Angles have to be floats or integers"

    jacobian = np.ones(shape=(2,2), dtype=float)

    jacobian[0][0] = -(l1 * np.sin(q0+q1) + l0*np.sin(q0))
    jacobian[0][1] = -l1*np.sin(q0+q1)
    jacobian[1][0] = l1 * np.cos(q0+q1) + l0*np.cos(q0)
    jacobian[1][1] = l1*np.cos(q0+q1)

    return jacobian


def getIK(delta_x, delta_y,
          q0_current=0, q1_current=0):
    """
    Inverse kinematics (using Jacobian Inverse)

    :param delta_x: Error in position (x)
    :param delta_y:
    :param q0_current: Current Joint angle 0
    :param q1_current: Current Joint angle 1
    :return: Joint angles
    """
    assert all(isinstance(i, (int, float)) for i in (delta_x, delta_y)), "Positions have to be floats or integers"
    assert all(isinstance(i, (int, float)) for i in (q0_current, q1_current)), "Angles have to be floats or integers"

    inverse_jacobian = np.linalg.pinv(getJacobian(q0_current, q1_current))
    assert inverse_jacobian.shape == (2, 2), f"{inverse_jacobian.shape}"

    delta_position = np.array([delta_x, delta_y]).reshape(-1, 1)
    delta_q = inverse_jacobian@delta_position
    assert delta_q.shape == (2, 1)

    return delta_q


def get_torque_end_effector_position(q0_obs, q1_obs,
                                     q0_dot_obs, q1_dot_obs,
                                     x_desired, y_desired,
                                     vx_ref=0, vy_ref=0,
                                     *args):
    """
    Given Current observation and PD controller terms, return Torque (action for env action space)
    :param vx_ref: Desired velocity for Derivative control, Default to zero: to try to get robot to rest after each step
    :param vy_ref: ""
    :param x_desired: Desired position (x)
    :param y_desired:
    :param q0_obs: Joint angle 0 (read from sensor)
    :param q1_obs:
    :param q0_dot_obs: Joint angle velocity 0 (read from sensor)
    :param q1_dot_obs:
    :param args: kp_1, kp_2, kd_1, kd_2 (PD Controller Gains)
    :return: joint_torques(action)
    """

    kp_1, kp_2, kd_1, kd_2 = args

    assert isinstance(vx_ref,(int, float)) and isinstance(vy_ref, (int, float))
    assert all(isinstance(i, (float, int))
               for i in (kp_1, kp_2, kd_1, kd_2)), "PD controller gains should be integer, float"

    KP = np.diag([kp_1, kp_2])
    KD = np.diag([kd_1, kd_2])

    jacobian = getJacobian(q0_obs,q1_obs)
    x_curr, y_curr = getForwardModel(q0_obs, q1_obs)[:2]
    e = np.array([x_desired - x_curr.item(), y_desired - y_curr.item()]).reshape(-1, 1)  # Error in position

    vx_curr, vy_curr = jacobian @ np.array([q0_dot_obs, q1_dot_obs]).reshape(-1, 1)   # dx/dt = J dq/dt
    V_e = np.array([vx_ref - vx_curr.item(), vy_ref - vy_curr.item()]).reshape(-1, 1)  # Error in velocity

    assert e.shape == (2,1) and V_e.shape == (2,1) , f"{e.shape}, {V_e.shape}"

    force_end_eff = KP @ e + KD @ V_e  # PD Controller
    joint_torques = (jacobian.T @ force_end_eff).reshape(-1)  # Torque= J.T @ F

    assert joint_torques.shape == (2,), f"{joint_torques.shape}"
    return joint_torques


def get_torque_joint_angle(q0_obs, q1_obs,
                           q0_dot_obs, q1_dot_obs,
                           x_desired, y_desired,
                           q0_dot_ref, q1_dot_ref,
                           *args):
    """
    Given Current Velocity and Angle (observation) and PD controller terms, return Torque (action for env action space)
    :param x_desired: Desired position (x)
    :param y_desired:
    :param q0_dot_ref: Needed angular velocity Ref (Same as vx_ref, vy_ref in get_torque_end_effector_position function
    :param q1_dot_ref:
    :param q0_obs: Joint angle 0 (read from sensor)
    :param q1_obs:
    :param q0_dot_obs: Joint angle velocity 0 (read from sensor)
    :param q1_dot_obs:
    :param args: kp_1, kp_2, kd_1, kd_2 (PD Controller Gains)
    :return: joint_torques(action)
    """

    kp_1, kp_2, kd_1, kd_2 = args

    assert all(isinstance(i, (float, int))
               for i in (q0_obs, q1_obs, q0_dot_obs, q1_dot_obs, q0_dot_ref)),\
        "Observations and reference should be float/int"
    assert all(isinstance(i, (float, int))
               for i in (kp_1, kp_2, kd_1, kd_2)), "PD controller gains should be integer, float"

    KP = np.diag([kp_1, kp_2])
    KD = np.diag([kd_1, kd_2])

    # Use Inverse Kinematics to get desired joint angle
    x_curr, y_curr = getForwardModel(q0_obs, q1_obs)[:2]
    delta_x, delta_y = x_desired - x_curr, y_desired - y_curr
    delta_q0, delta_q1 = getIK(delta_x, delta_y, q0_current=q0_obs, q1_current=q1_obs)

    e = np.array([delta_q0, delta_q1]).reshape(-1, 1)
    T_e = np.array([q0_dot_ref - q0_dot_obs, q1_dot_ref - q1_dot_obs]).reshape(-1, 1)  # Error in angular velocity

    assert e.shape == (2,1) and T_e.shape == (2, 1), f"{e.shape}, {T_e.shape}"
    joint_torques = KP @ e + KD @ T_e  # PD Controller
    joint_torques = joint_torques.reshape(-1)

    assert joint_torques.shape == (2,), f"{joint_torques.shape}"
    return joint_torques


def get_samples_from_trajectory(steps =100):
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


def create_trajectory_general(steps=100,
                      kp_1=1.0, kp_2=1.0, kd_1=1.0, kd_2=1.0,
                      q0_curr=-np.pi, q1_curr=-np.pi,
                      episodes=1, input_signal="end_effector_position"):
    """
    Given steps, Proportion Control(kp_1, kp_2) and Differential Control parameters (kd_1, kd_2) return observed trajectory (x_obs, y_obs)
    :param q0_curr: Starting position Joint 0
    :param q1_curr: Starting position Joint 1
    :param episodes: Number of sequences (one sequence is, one complete lap along the trajectory)
    :param steps: Number of discretized steps
    :param kp_1: Proportion Control [0][0]
    :param kp_2: Proportion Control [1][1]
    :param kd_1: Derivative Control [0][0]
    :param kd_2: Derivative Control [1][1]
    :return: obs_trajectory : (2, steps) numpy array
    """
    assert isinstance(steps, int), "steps has to be integer"
    assert all(isinstance(i, (float, int)) for i in (kp_1, kp_2, kd_1, kd_2)),"PD controller gains should be integer, float"
    assert isinstance(input_signal, str)

    env = gym.make("ReacherPyBulletEnv-v0")

    # env.render()
    env.reset()
    desired_traj = get_samples_from_trajectory(steps)
    final_trajectory = np.zeros(shape=(2, steps), dtype=float)

    for curr_episode in range(episodes):  # For multiple episodes, Default: episodes= 1
        # Set robot to starting spot and record starting point in trajectory
        env.unwrapped.robot.central_joint.reset_position(q0_curr, 0)
        env.unwrapped.robot.elbow_joint.reset_position(q1_curr, 0)
        final_trajectory[:, 0] = getForwardModel(q0_curr, q1_curr)[:2]

        q0_obs, q1_obs = q0_curr, q1_curr
        q0_dot_obs, q1_dot_obs = 0, 0

        for robo_step in range(steps-1):
            x_desired = desired_traj[0, robo_step+1]
            y_desired = desired_traj[1, robo_step+1]

            # action = env.action_space.sample() #[0.5, 0.7] Sample action (Torque) for q0, q1
            if input_signal == "end_effector_position":
                vx_ref, vy_ref = 0, 0
                action = get_torque_end_effector_position(q0_obs, q1_obs,
                                                          q0_dot_obs, q1_dot_obs,
                                                          x_desired, y_desired,
                                                          vx_ref, vy_ref,
                                                          kp_1, kp_2, kd_1, kd_2)
            else:
                q0_dot_ref, q1_dot_ref = 0, 0
                action = get_torque_joint_angle(q0_obs, q1_obs,
                                                q0_dot_obs, q1_dot_obs,
                                                x_desired, y_desired,
                                                q0_dot_ref, q1_dot_ref,
                                                kp_1, kp_2, kd_1, kd_2)

            _ = env.step(action)  # Provide Torque to Robot

            q0_obs, q0_dot_obs = env.unwrapped.robot.central_joint.current_position()  # Current Observation from Sensor
            q1_obs, q1_dot_obs = env.unwrapped.robot.elbow_joint.current_position()

            final_trajectory[:, robo_step+1] = getForwardModel(q0_obs, q1_obs)[:2]  # Current trajectory x

    env.close()

    return final_trajectory


def plot_trajectory(desired_traj, final_trajectory, title):
    """
    Plot Trajectory
    :param final_trajectory:  Trajectory to be plotted, numpy array (2, steps)
    :param desired_traj:
    :param title: title of plot
    :return: None
    """
    assert isinstance(desired_traj, np.ndarray) and desired_traj.shape[0] == 2
    assert isinstance(final_trajectory, np.ndarray) and final_trajectory.shape[0] == 2

    plt.plot(desired_trajectory[0, :], desired_trajectory[1, :], "r-", linewidth=2, label='Desired Trajectory')
    # plt.plot(final_trajectory[0, :], final_trajectory[1, :], "g-", linewidth=2, label='Final Trajectory')

    # # Single points
    # plt.scatter(final_trajectory[0, 0], final_trajectory[1, 0], c='g', marker='+', linewidths=1)
    # plt.scatter(final_trajectory[0, -1], final_trajectory[1, -1], c='g', marker='+', linewidths=1)

    # # All points
    # plt.scatter(desired_trajectory[0, :], desired_trajectory[1, :], c='r', marker='.', label='Desired Trajectory')
    plt.scatter(final_trajectory[0, :], final_trajectory[1, :], c='g', marker='.', label='Final Trajectory')

    plt.legend(loc='upper left', prop={'size': 6})
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    steps = 5000
    q0_curr, q1_curr = -np.pi, 0.0  # Start at leftmost tip of the figure [ Trajectory(-np.pi, -np.pi) ]
    pd_controller_gains = (4.5, 4.5, 0.5, 0.5)

    input_signals = {1: "end_effector_position",
                     2: "joint_angle"}

    inp = 1  # For choosing the control (from above)

    desired_trajectory = get_samples_from_trajectory(steps=steps)
    final_trajectory = create_trajectory_general(steps,  *pd_controller_gains,
                                                 q0_curr, q1_curr,
                                                 input_signal=input_signals[inp])

    mse = np.mean(sum(desired_trajectory - final_trajectory)**2)
    print("MSE between trajectories:", mse.item())

    plot_trajectory(desired_trajectory, final_trajectory, title=input_signals[inp])


