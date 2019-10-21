import numpy as np
import matplotlib.pyplot as plt
import gym
import pybulletgym.envs

import random


def TestPolicy(policy=lambda s: (s+1) % 4 , render= True, allowed_steps = 200):
    """
    Given deterministic policy return average rate of successful episodes over 100 trials
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
            steps +=1
            obs, reward, done, info = env.step()
            if render: env.render()
            done = done and (steps < allowed_steps)  # Terminate when Reach Goal/ Reach Hole / Ran out of steps

        net_success_rate += reward
        # If the latest reward=1, you reached the goal and loop was terminated, else reward 0

    average_success_rate = net_success_rate/100.0

    return average_success_rate




if __name__ == "__main__":

    env = gym.make("FrozenLake-v0")

    episodes = 1
    done = False
    for episode in range(episodes):
        state = env.reset()
        steps = 0
        while not done:
            print(f"State:{state} Done:{done} Steps:{steps}")
            action = random.randint(0, 3)  # is action 1,4
            steps += 1
            state, reward, done, info = env.step(action)
            print(f"Action:{action} Reward:{reward}")
            env.render()





