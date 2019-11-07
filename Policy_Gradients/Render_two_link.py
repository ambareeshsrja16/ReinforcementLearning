
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import math

class PolicyNet(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden=64, lr=0.001):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)
        self.sigma = torch.nn.parameter.Parameter(torch.tensor([[0.01, 0], [0, 0.01]]))  # 1
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def getAction(self, state):
        # implement get action
        mu = self.forward(state)
        eps = 1e-3 * torch.tensor([[1.0, 0], [0, 1.0]])  # 1
        if self.training:
            m = MultivariateNormal(mu, torch.abs(self.sigma) + eps)  # 1
        else:
            return mu, 0

        try:
            action = m.sample()
        except:
            print(state, mu, self.sigma, m)
            raise
        return action, m.log_prob(action)

def getDiscountedFutureReturns(rewards, b=False, gamma=0.9):
    future_rewards = []
    last_reward = 0
    for r in rewards[::-1]:
        future_rewards.append(r + gamma * last_reward)
        last_reward = future_rewards[-1]
    future_rewards = future_rewards[::-1]
    return future_rewards

def updatePolicy2(policyNet, future_returns, log_probs, n_episodes):
    if n_episodes == 0:
        n_episodes = 1

    future_returns = torch.FloatTensor(future_returns)
    future_returns = (future_returns - future_returns.mean()) / (future_returns.std() + 1e-7)

    policyGradient = []
    for log_prob, G in zip(log_probs, future_returns):
        policyGradient.append(-log_prob * G)

    policyNet.optimizer.zero_grad()
    policyGradient = torch.stack(policyGradient).sum() / n_episodes
    policyGradient.backward()
    policyNet.optimizer.step()

def reinforce(b=False, bs=2000, iteratons=500):
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    env.reset()
    policyNet = PolicyNet(env.observation_space.shape[0] - 1, env.action_space.shape[0], n_hidden=64, lr=lr)

    itrs = iteratons
    maxSteps = bs
    avg_returns = []

    for itr in range(itrs):
        state = env.reset()

        episodes = 0
        rewards_sum, log_prob_sum = 0, 0
        G, log_probs, rewards = [], [], []
        log_probs_episode, total_ll_episodes = [], []
        all_future_returns, all_log_probs = [], []

        for step in range(maxSteps):
            # get action
            state = torch.tensor(state).float()
            action, log_prob = policyNet.getAction(state)

            # take step and measure state, reward
            newState, reward, done, _ = env.step(action.numpy())
            rewards.append(reward)
            log_probs_episode.append(log_prob)
            state = newState

            # when an episode ends
            if done or step == maxSteps - 1:
                episodes += 1
                state = env.reset()
                future_rewards = getDiscountedFutureReturns(rewards, b)
                rewards_sum += sum(rewards)

                assert len(future_rewards) == len(log_probs_episode)
                all_future_returns.extend(future_rewards)
                all_log_probs.extend(log_probs_episode)

                rewards, log_probs_episode = [], []

        updatePolicy2(policyNet, all_future_returns, all_log_probs, episodes)
        avg_returns.append(rewards_sum / episodes)
        if itr % 10 == 0:
            print('avg return for ', itr, avg_returns[-1])
            print(policyNet.sigma)
        if itr % 20 == 0:
            plt.plot(avg_returns)
            plt.title('Avg return vs iterations, bs = ' + str(bs))
            plt.show()
            plt.figure()
            plt.close()


        G, rewards, log_probs = [], [], []

    plt.plot(avg_returns)
    plt.title('Avg return vs iterations, bs = ' + str(bs))
    plt.close()
    env.close()
    return policyNet, avg_returns

def testPolicy(policyNet):
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    env.render(mode="human")
    state = env.reset()
    done = False
    total_reward = 0
    for i in range(1000):
        state = torch.tensor(state).float()
        action, log_prob = policyNet.getAction(state)
        state, reward, done, _ = env.step(action.numpy())
        env.render()
        total_reward += reward
        time.sleep(0.2)
        if done:
            print('steps: ', i + 1)
            break
    print(total_reward)
    env.close()

if __name__ == '__main__':
    gamma = 0.9
    lr = 0.001
    policyNet, avg_returns = reinforce(b=True, bs=2000, iteratons=500)
    testPolicy(policyNet)





