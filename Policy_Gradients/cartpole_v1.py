import numpy as np
import gym
import pybulletgym.envs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch.optim


class PolicyNetwork(nn.Module):

    def __init__(self, units=6, activation="Relu"):
        super(PolicyNetwork, self).__init__()

        self.activation = activation
        self.fc1 = nn.Linear(4, units)
        self.fc2 = nn.Linear(units, 2)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):  # Convert numpy array to torch Tensor
            state = torch.from_numpy(state).float()  # Model parameters are in float, numpy by default goes with Double
        if self.activation == "Relu":
            probabilities = F.softmax(self.fc2(F.relu(self.fc1(state))), dim = 0)

        return probabilities


class CartPole:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.policy_network = PolicyNetwork()

    def do_reinforce(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=0.01):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """

        average_returns = [0]*iterations

        cov_scaling_factor = np.random.rand()  # DIKEO should we tune this?
        covariance = cov_scaling_factor * torch.eye(4)  # Covariance matrix initialized to a random value
        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=learning_rate)

        for iteration in range(iterations):  # Updated once per loop

            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0
            prev_state = self.env.reset()
            current_trajectory = {"log_probabilities": [0]*batch_size,
                                       "rewards": [0]*batch_size}
            while step < batch_size:  # collection of data for trajectories

                # DIKEO Check this?
                probabilities = self.policy_network(prev_state)  # Forward pass

                assert probabilities.shape == (2,), f"{probabilities.shape}"
                assert 0.99 <= sum(probabilities) <= 1.01, f"{sum(probabilities)}"

                probable_actions = torch.distributions.Categorical(probabilities)
                # DIKEO presumably 2 element vector of associated probabilities

                action = probable_actions.sample()  # Choose 0 or 1
                assert action.item() in (0, 1), f"{action}"

                current_state, reward, done, _ = self.env.step(action.item())

                current_trajectory["log_probabilities"][step] = probable_actions.log_prob(action)
                current_trajectory["rewards"][step] = reward

                prev_state = current_state
                if done:
                    prev_state = self.env.reset()
                step += 1

            # DIKEO
            # How to update the covariance matrix?
            # DIKEO
            # Assert for covariance matrix eigenvalues to be >0 and <0.001
            # G(T)
            # DIKEO Is this always calculated from the start?

            discounted_reward = sum(current_trajectory["rewards"][step] * gamma**step
                                    for step in range(batch_size))
            sum_of_log_prob = sum(current_trajectory["log_probabilities"])

            average_returns[iteration] = discounted_reward

            # DIKEO Is the loss correct?
            loss = -1 * discounted_reward * sum_of_log_prob
            loss.backward()
            optimizer.step()
            print(f"Iteration: {iteration}, Discounted Reward (G(tau)): {discounted_reward}")

        self.plot_average_returns(average_returns)

    @staticmethod
    def plot_average_returns(returns, title ="Average Returns"):
        """

        :param returns:
        :return:
        """

        fig, ax = plt.subplots()
        ax.plot(returns)
        plt.title(title)
        plt.ylabel("Iterations")
        plt.show()

if __name__ == "__main__":
    save_path = "/Users/ambareeshsnjayakumari/Desktop/Tabular_Methods"
    bot = CartPole()
    bot.do_reinforce(iterations=10, batch_size=500)

