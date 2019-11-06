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

    def __init__(self, units):
        super(PolicyNetwork, self).__init__()

        # DIKEO Change the Neural network
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 150)
        self.fc3 = nn.Linear(150, 200)
        self.fc4 = nn.Linear(200, 2)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):  # Convert numpy array to torch Tensor
            state = torch.from_numpy(state).float()  # Model parameters are in float, numpy by default goes with Double

        # probabilities = F.tanh(self.fc2(F.tanh(self.fc1(state))), dim=0)

        # DIKEO: Using tanh for the final layer, is that correct? Limits of action space match with tanh limits
        mean = torch.tanh(self.fc4(torch.tanh(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(state))))))))

        return mean


class TwoLinkArm:

    def __init__(self, units):
        self.env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
        # initializes environment at a fixed position
        # DIKEO , how to get that fixed position? And why can't I start from any random position?
        # how does that affect training?

        self.policy_network = PolicyNetwork(units)

    def do_reinforce(self, iterations=500, batch_size=500, gamma=0.9, learning_rate=0.01, covariance=0.01):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """

        # Initialize covariance matrix with random values. Here a diagonal with values 0.01
        covariance = torch.from_numpy(np.diag(covariance*np.ones(2))).float()

        average_returns = [0]*iterations
        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=learning_rate)
        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0

            prev_state = self.env.reset()
            # assert prev_state.shape == self.env.observation_space.shape, f"{prev_state.shape}"

            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                mean = self.policy_network(prev_state)  # Forward pass,
                # Neural network returns the mean of the MVGaussian distribution for action
                # Sampling to be done in continous space

                assert mean.shape == (2,), f"{mean.shape}"
                assert -2.00 <= sum(mean) <= 2.00, f"{sum(mean)}"

                probable_actions = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance)
                action = probable_actions.sample()  # Choose a 2-D vector for action

                assert action.size() == (2,), f"{action, action.shape}"

                current_state, reward, done, _ = self.env.step(action)

                current_trajectory["log_probabilities"].append(probable_actions.log_prob(action))
                current_trajectory["rewards"].append(reward)

                prev_state = current_state

                if done:  # episode terminated
                    discounted_reward = sum(current_trajectory["rewards"][t] * (gamma**t) for t in range(len(current_trajectory["rewards"])))
                    # calculate discounted reward for trajectory
                    sum_of_log_prob = sum(current_trajectory["log_probabilities"])
                    loss += discounted_reward * sum_of_log_prob
                    net_reward += discounted_reward

                    current_trajectory = {"log_probabilities": [], "rewards": []}  # Resetting for collecting next episode data
                    prev_state = self.env.reset()
                    episodes += 1

                step += 1

            # End of 500 steps, time to update policy
            loss = -1 * loss * (1/episodes)  # Objective function, -1 => Gradient Ascent (Torch does descent by default)

            average_returns[iteration] = net_reward/episodes  # For plotting later

            loss.backward()  # Calculate gradient
            optimizer.step()  # Update neural net (aka policy)

            print(f"Iteration: {iteration}, Average Return (G(tau)/episodes): {average_returns[iteration]}")

        self.plot_average_returns(average_returns)

    def do_reinforce_modified_policy_gradient(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=0.01, covariance=0.01):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        # Initialize covariance matrix with random values. Here a diagonal with values 0.01
        covariance = torch.from_numpy(np.diag(covariance * np.ones(2))).float()

        average_returns = [0] * iterations
        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=learning_rate)
        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0

            prev_state = self.env.reset()
            # assert prev_state.shape == self.env.observation_space.shape, f"{prev_state.shape}"

            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                mean = self.policy_network(prev_state)  # Forward pass,
                # Neural network returns the mean of the MVGaussian distribution for action
                # Sampling to be done in continous space

                assert mean.shape == (2,), f"{mean.shape}"
                assert -2.00 <= sum(mean) <= 2.00, f"{sum(mean)}"

                probable_actions = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance)
                action = probable_actions.sample()  # Choose a 2-D vector for action

                assert action.size() == (2,), f"{action, action.shape}"

                current_state, reward, done, _ = self.env.step(action)

                current_trajectory["log_probabilities"].append(probable_actions.log_prob(action))
                current_trajectory["rewards"].append(reward)

                prev_state = current_state

                if done:  # episode terminated
                    T = len(current_trajectory["rewards"]) # Total steps
                    discounted_reward = sum(current_trajectory["rewards"][t] * (gamma**t) for t in range(len(current_trajectory["rewards"])))

                    objective_over_all_episodes = [current_trajectory["log_probabilities"][t] * sum([(gamma**(t_bar-t)) * current_trajectory["rewards"][t_bar]])
                                                    for t in range(T)
                                                    for t_bar in range(t, T)]

                    objective_over_all_episodes = sum(objective_over_all_episodes)

                    # # # OR
                    # objective_over_all_episodes = 0.0
                    # for t in range(T):
                    #     summed_future_reward = 0.0
                    #     for t_bar in range(t, T):
                    #         summed_future_reward += (gamma**(t_bar-t)) * current_trajectory["rewards"][t_bar]
                    #     objective_over_all_episodes += current_trajectory["log_probabilities"][t] * summed_future_reward
                    # # #

                    loss += objective_over_all_episodes

                    net_reward += discounted_reward

                    current_trajectory = {"log_probabilities": [], "rewards": []}  # Resetting for collecting next episode data
                    prev_state = self.env.reset()
                    episodes += 1

                step += 1

            # End of 500 steps, time to update policy
            loss = -1 * loss * (1/episodes)  # Objective function, -1 => Gradient Ascent (Torch does descent by default)

            average_returns[iteration] = net_reward/episodes  # For plotting later

            loss.backward()  # Calculate gradient
            optimizer.step()  # Update neural net (aka policy)

            print(f"Iteration: {iteration}, Average Return (G(tau)/episodes): {average_returns[iteration]}")

        self.plot_average_returns(average_returns)

    def do_reinforce_modified_policy_gradient_bias_subtracted(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=0.01, covariance=0.01, b=15.0):
        """
        DIKEO decide b ?
        b
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        # Initialize covariance matrix with random values. Here a diagonal with values 0.01
        covariance = torch.from_numpy(np.diag(covariance * np.ones(2))).float()

        average_returns = [0] * iterations
        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=learning_rate)
        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0

            prev_state = self.env.reset()
            # assert prev_state.shape == self.env.observation_space.shape, f"{prev_state.shape}"

            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                mean = self.policy_network(prev_state)  # Forward pass,
                # Neural network returns the mean of the MVGaussian distribution for action
                # Sampling to be done in continous space

                assert mean.shape == (2,), f"{mean.shape}"
                assert -2.00 <= sum(mean) <= 2.00, f"{sum(mean)}"

                probable_actions = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance)
                action = probable_actions.sample()  # Choose a 2-D vector for action

                assert action.size() == (2,), f"{action, action.shape}"

                current_state, reward, done, _ = self.env.step(action)

                current_trajectory["log_probabilities"].append(probable_actions.log_prob(action))
                current_trajectory["rewards"].append(reward)

                prev_state = current_state

                if done:  # episode terminated
                    T = len(current_trajectory["rewards"])  # Total steps
                    discounted_reward = sum((current_trajectory["rewards"][t]) * (gamma ** t) for t in
                                            range(len(current_trajectory["rewards"])))

                    objective_over_all_episodes = [current_trajectory["log_probabilities"][t] * sum([(gamma ** (t_bar - t)) * (current_trajectory["rewards"][t_bar]-b)])
                                                   for t in range(T)
                                                   for t_bar in range(t, T)]

                    objective_over_all_episodes = sum(objective_over_all_episodes)

                    # # # OR
                    # objective_over_all_episodes = 0.0
                    # for t in range(T):
                    #     summed_future_reward = 0.0
                    #     for t_bar in range(t, T):
                    #         summed_future_reward += (gamma**(t_bar-t)) * current_trajectory["rewards"][t_bar]
                    #     objective_over_all_episodes += current_trajectory["log_probabilities"][t] * summed_future_reward
                    # # #

                    loss += objective_over_all_episodes

                    net_reward += discounted_reward

                    current_trajectory = {"log_probabilities": [],
                                          "rewards": []}  # Resetting for collecting next episode data
                    prev_state = self.env.reset()
                    episodes += 1

                # print(episodes)
                step += 1

            # End of 500 steps, time to update policy
            loss = -1 * loss * (
                        1 / episodes)  # Objective function, -1 => Gradient Ascent (Torch does descent by default)

            average_returns[iteration] = net_reward / episodes  # For plotting later

            loss.backward()  # Calculate gradient
            optimizer.step()  # Update neural net (aka policy)

            print(f"Iteration: {iteration}, Average Return (G(tau)/episodes): {average_returns[iteration]}")

        self.plot_average_returns(average_returns)

    @staticmethod
    def plot_average_returns(returns, title="Average Returns"):
        """
        Plots average returns
        """
        fig, ax = plt.subplots()
        ax.plot(returns)
        plt.title(title)
        plt.xlabel("Iterations")
        plt.show()


if __name__ == "__main__":
    save_path = "/Users/ambareeshsnjayakumari/Desktop/Policy_Gradients"
    reinforce_types = {1: TwoLinkArm.do_reinforce,
                       2: TwoLinkArm.do_reinforce_modified_policy_gradient,  # DIKEO Implement
                       3: TwoLinkArm.do_reinforce_modified_policy_gradient_bias_subtracted}  # DIKEO Implement

    bot = TwoLinkArm(64)
    required_function = reinforce_types[3]
    required_function(bot, iterations=500, batch_size=500, gamma=0.9, learning_rate=0.01, covariance=0.01)



