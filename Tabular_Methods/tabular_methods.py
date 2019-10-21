import numpy as np
import matplotlib.pyplot as plt
import gym
import pybulletgym.envs
import random


def LearnModel():
    """

    :return:
    """
    joint_occurrence_matrix_sbar_a_s = np.zeros((16, 4, 16))  # 16 states, 4 actions
    reward_matrix_sbar_a_s = np.zeros((16, 4, 16))  # 16 states, 4 actions

    prev_state = env.reset()
    for sample in range(10**5):  # 10^5 random samples
        action = env.action_space.sample()
        current_state, reward, done, _ = env.step(action)

        joint_occurrence_matrix_sbar_a_s[current_state, action, prev_state] += 1
        reward_matrix_sbar_a_s[current_state, action, prev_state] += 1

        prev_state = current_state

        if done:
            prev_state = env.reset()  # reset if episode terminates

    assert np.sum(joint_occurrence_matrix_sbar_a_s) == 10**5

    joint_prob_matrix_sbar_a_s = joint_occurrence_matrix_sbar_a_s/(10**5)
    # P(a,b,c)

    joint_prob_matrix_a_s = np.sum(joint_prob_matrix_sbar_a_s, axis=0)
    # P(b,c)

    conditional_prob_matrix_sbar_given_a_s = joint_prob_matrix_sbar_a_s/joint_prob_matrix_a_s  # Cross check

    np.nan_to_num(conditional_prob_matrix_sbar_given_a_s, copy=False, nan=0.0)  # change in place
    # P(a | b,c) = P(a,b,c)/ P(b,c)
    # Broadcasting
    # (16, 4, 16) / (4,16)
    # (16, 4, 16) / (1,4,16) Pad on left
    # (16,4, 16) / (16,4,16  Stretch

    reward_matrix_sbar_a_s /= 10**5  # Normalize

    return conditional_prob_matrix_sbar_given_a_s, reward_matrix_sbar_a_s


def TestPolicy(policy=lambda s: (s+1) % 4, render=False, allowed_steps=200):
    """
    Given deterministic policy return average rate of successful episodes over 100 trials
    :param render:
    :param allowed_steps:
    :param policy:
    :return:
    """

    assert hasattr(policy, "__call__"),  "policy has to be a function"

    net_success_rate = 0
    for episode in range(100):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = policy(state)  # deterministic
            steps += 1
            state, reward, done, info = env.step(action)
            if render:
                env.render()
            done = done and (steps < allowed_steps)  # Terminate when Reach Goal/ Reach Hole / Ran out of steps

        net_success_rate += reward
        # If the latest reward=1, you reached the goal and loop was terminated, else reward 0

    average_success_rate = net_success_rate  # Success Rate %

    return average_success_rate


if __name__ == "__main__":

    env = gym.make("FrozenLake-v0")

    success_rate = TestPolicy()
    transition_prob_matrix, reward_matrix = LearnModel()
    print(np.sum(transition_prob_matrix))
    print(np.sum(reward_matrix))






