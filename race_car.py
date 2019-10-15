import numpy as np
import matplotlib.pyplot as plt
import gym
import pybulletgym.envs


def get_angle_thrust(*args):
    """
    kp_1, kp_2, kd_1, kd_2, kw_1 = args[:5]
    delta_x, delta_y, delta_vx, delta_vy, delta_w = args[5:]
    :return:
    """
    kp, kd = args[:2]
    delta_x, delta_y, v, theta = args[2:]

    assert all(isinstance(i, (float, int))for i in args), f"Observations, PD controller gains should be float/int \n {args}"

    # Bicycle model. The point you want to get to is at delta_y, delta_x angle from your current point
    new_theta = np.arctan2(delta_y, delta_x)

    wheel_angle = new_theta - theta

    # Correcting for wrap-around of wheel angle
    # If the wheel angle is outside the range of [-pi, pi] bring it back to the range by adding or subtracting 2*pi
    #   Notice that the wheel angle falls outside the range of [-pi, pi],
    #       when the new_theta and theta are in different quadrants (some specific quadrants)
    # The while loop is because, sometimes the difference evaluates to angles like 690 (degrees),
    #   so you have to subtract twice
    if wheel_angle <= -np.pi:
        while not -np.pi <= wheel_angle <= np.pi:
            wheel_angle += 2 * np.pi
    elif wheel_angle >= np.pi:
        while not -np.pi <= wheel_angle <= np.pi:
            wheel_angle -= 2 * np.pi

    # # DEBUG
    # print("dy, dx", delta_y, delta_x)
    # print("\nDesired Theta, Current Theta, Steering Angle (All in Degrees)")
    # print(np.rad2deg([new_theta, theta, wheel_angle]))

    # Normalize between -1,1
    #   (The action space of the wheel angle of car is [-1,1] mapped from actual range [-pi/2,pi/2])
    wheel_angle = ((wheel_angle - (-np.pi/2))/np.pi)*2 - 1  # (x-x_min)/(x_max- x_min)-> [0,1] *2 -> [0,2]-> -1 ->[-1,1]

    e = np.sqrt((delta_x**2 + delta_y**2)) # Error in position
    v_desired = np.array([np.sqrt((delta_x**2 + delta_y**2) / (np.cos(wheel_angle))**2)])
    V_e = v_desired - v  # Error in linear velocity

    thrust = kp*e + kd*V_e  # PD Controller
    thrust = thrust.item()

    # Clipping if it goes beyond limits, normalize between -1,1
    #   (The action space of the thrust of car is [-1,1] mapped from actual range [0,20])
    thrust = np.clip(thrust, 0, 20).item()
    # Normalizing thrust to [-1,1]
    thrust = (thrust/20)*2 - 1

    return wheel_angle, thrust


def create_trajectory_general(*args, input_signal="Circle"):
    """
    Given steps, Proportion Control(kp_1, kp_2) and Differential Control parameters (kd_1, kd_2) return observed trajectory (x_obs, y_obs)
    :param input_signal:
    :param kw_1:
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
    assert all(isinstance(i, (float, int)) for i in args), "PD controller gains should be integer, float"
    assert isinstance(input_signal, str)

    from racecar.SDRaceCar import SDRaceCar
    env = SDRaceCar(render_env=False, track=input_signal)

    desired_trajectory = []
    final_trajectory = []

    state = env.reset()
    x, y = state[-1]
    theta = state[2]
    vx, vy = state[3], state[4]
    x_desired, y_desired = state[-1]  # next desired position, from track

    env.x, env.y = x, y # SET CAR TO START POSITION?
    desired_trajectory.append(np.array([x_desired, y_desired]))
    final_trajectory.append(np.array([x, y]))  # x, y

    previous_ind = 0
    steps = 0
    done = False

    while not done:
        delta_x, delta_y = x_desired-x, y_desired-y
        v = np.sqrt(vx**2 + vy**2).item()

        action = get_angle_thrust(*args,
                                  delta_x, delta_y, v, theta)

        state, r, done = env.step(action)
        x, y = state[:2]
        theta = state[2]
        vx, vy = state[3:5]
        x_desired, y_desired = state[-1]  # next desired position, from track

        desired_trajectory.append(state[-1])  # next desired position from track (Read from sensor)
        final_trajectory.append(np.array([state[0], state[1]]))  # Current position, (Read from sensor)

        steps += 1
        current_ind = env.closest_track_ind

        # CONDITION TO CHECK lap-completion
        if current_ind - previous_ind <= -500:
            done = True
            # # DEBUG
            # print("Steps (DONE):", steps)
        previous_ind = current_ind

    print("Steps(RETURN):", steps)
    return desired_trajectory, final_trajectory, steps


def plot_trajectory(desired_trajectory, final_trajectory, title):
    """
    Plot Trajectory
    :param final_trajectory:  Trajectory to be plotted, numpy array (2, steps)
    :param desired_traj:
    :param title: title of plot
    :return: None
    """
    assert isinstance(desired_trajectory, np.ndarray) and desired_trajectory.shape[0] == 2
    assert isinstance(final_trajectory, np.ndarray) and final_trajectory.shape[0] == 2

    plt.ioff()

    # # Plot trajectory
    plt.plot(desired_trajectory[0, :], desired_trajectory[1, :], "r-", linewidth=2, label='Desired Trajectory')
    # plt.plot(final_trajectory[0, :], final_trajectory[1, :], "g-", linewidth=2, label='Final Trajectory')

    # # Single point
    # plt.scatter(desired_trajectory[0, 0], desired_trajectory[1, 0], c='y', marker='o', label='Desired Trajectory Start')
    # plt.scatter(final_trajectory[0, 0], final_trajectory[1, 0], c='b', marker='o', label='Final Trajectory Start')

    # Few points
    # until = 50  # until k steps
    # plt.scatter(desired_trajectory[0, :until], desired_trajectory[1, :until], c='r', marker='.', label='Desired Trajectory')
    # plt.scatter(final_trajectory[0, :until], final_trajectory[1, :until], c='g', marker='.', label='Final Trajectory')

    # # All points
    # plt.scatter(desired_trajectory[0, :], desired_trajectory[1, :], c='r', marker='.', label='Desired Trajectory')
    plt.scatter(final_trajectory[0, :], final_trajectory[1, :], c='g', marker='.', label='Final Trajectory')

    plt.legend(loc='best', prop={'size': 6})
    plt.title(title)
    plt.show()


def get_tuned_trajectory(*pd_controller_gains, input_number):
    """
    :param input_number:
    :param pd_controller_gains:
    :return:
    """
    input_signals = {1: "Linear",
                     2: "FigureEight",
                     3: "Circle"}

    desired_trajectory_list, final_trajectory_list, steps = create_trajectory_general(
        *pd_controller_gains,
        input_signal=input_signals[input_number])

    desired_trajectory = np.zeros((2, len(desired_trajectory_list)))
    final_trajectory = np.zeros((2, len(desired_trajectory_list)))

    for i in range(len(desired_trajectory_list)):
        desired_trajectory[0, i], desired_trajectory[1, i] = desired_trajectory_list[i]
        final_trajectory[0, i], final_trajectory[1, i] = final_trajectory_list[i]

    mse = np.mean(sum(desired_trajectory - final_trajectory) ** 2)
    print("MSE between trajectories:", mse.item())
    print(f"Steps taken for {input_signals[input_number]}: {steps}")

    plot_trajectory(desired_trajectory, final_trajectory, title=input_signals[input_number]+f" | Steps: {steps}"+f" | MSE: {mse:.2f}")


if __name__ == "__main__":

    pd_controller_gains = (1.3, 0.5)
    input_signals = {1: "Linear",
                     2: "FigureEight",
                     3: "Circle"}

    needed_figures = (1, 2, 3)
    for i in needed_figures:
        get_tuned_trajectory(*pd_controller_gains, input_number=i)

# TO DO : print number of steps in figure