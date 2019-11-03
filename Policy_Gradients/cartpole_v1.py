import numpy as np
import gym
import pybulletgym.envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch.optim

class Policy_Network(nn.Module):

    def __init__(self, units=4, activation="relu"):
        super(Policy_Network, self).__init__()

        self.activation = activation
        self.fc1 = nn.Linear(4, units)
        self.fc2 = nn.Linear(units, 4)

    def forward_pass(self, state):
        if self.activation == "relu":
            mu = F.relu(self.fc2(F.relu(self.fc1(state))))
        return mu


class CartPole:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.iterations = 200
        self.batch_size = 500

        self.gamma = 0.99

        self.policy_network = Policy_Network()
        self.learning_rate = 0.01

    def do_reinforce(self):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        cov_scaling_factor = np.random.rand()  # DIKEO should we tune this?
        covariance = cov_scaling_factor * torch.eye(4)  # Covariance matrix initialized to a random value

        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=self.learning_rate)

        for update in range(self.iterations):  # Updated once per loop

            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0
            prev_state = self.env.reset()
            current_trajectory = {"log_probabilities": [0]*self.batch_size,
                                       "rewards": [0]*self.batch_size}
            while step < self.batch_size:  # collection of data for trajectories
                # DIKEO Check this?
                mean = self.policy_network(prev_state)  # Forward pass
                assert mean.shape == (2,)
                probs = torch.distributions.multinomial.Multinomial(mean, covariance)
                multinomial = torch.distributions.Categorical(probs)  # DIKEO presumably 2 element vector for 2 actions/ associated probabilities
                action = multinomial.sample()

                current_state, reward, done, _ = self.env.step(action)
                # DIKEO, what to do if you get stuck in done?

                current_trajectory["log_probabilities"][step] = multinomial.log_prob(action)
                current_trajectory["rewards"][step] = reward

                step += 1

            # DIKEO
            # How to update the covariance matrix?

            # DIKEO
            # Assert for covariance matrix eigenvalues to be >0 and <0.001

            # G(T)
            # DIKEO Is this always calculated from the start?
            discounted_reward = sum(current_trajectory["rewards"][step] * self.gamma**step
                                    for step in range(self.batch_size))
            sum_of_log_prob = sum(current_trajectory["log_probabilities"])

            # DIKEO Is the loss correct?
            loss = -1 * discounted_reward * sum_of_log_prob
            loss.backward()
            optimizer.step()




if __name__ == "__main__":
    save_path = "/Users/ambareeshsnjayakumari/Desktop/Tabular_Methods"

    bot = CartPole()

