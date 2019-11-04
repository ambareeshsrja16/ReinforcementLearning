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

        # self.fc1 = nn.Linear(4, units)
        # self.fc2 = nn.Linear(units, 2)

        self.fc1 = nn.Linear(4, units//4)
        self.fc2 = nn.Linear(units//4, units//2)
        self.fc3 = nn.Linear(units//2, units//4)
        self.fc4 = nn.Linear(units//4, 2)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):  # Convert numpy array to torch Tensor
            state = torch.from_numpy(state).float()  # Model parameters are in float, numpy by default goes with Double

        # probabilities = F.softmax(self.fc2(F.relu(self.fc1(state))), dim=0)

        probabilities = F.softmax(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state))))))), dim=0)

        return probabilities


class CartPole:

    def __init__(self, units):
        self.env = gym.make("CartPole-v1")
        self.policy_network = PolicyNetwork(units)

    def do_reinforce(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=0.01):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        average_returns = [0]*iterations
        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=learning_rate)

        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0
            prev_state = self.env.reset()

            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                probabilities = self.policy_network(prev_state)  # Forward pass

                assert probabilities.shape == (2,), f"{probabilities.shape}"
                assert 0.999 <= sum(probabilities) <= 1.001, f"{sum(probabilities)}"

                probable_actions = torch.distributions.Categorical(probabilities)
                action = probable_actions.sample()  # Choose 0 or 1

                assert action.item() in (0, 1), f"{action}"

                current_state, reward, done, _ = self.env.step(action.item())

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

    def do_reinforce_modified_policy_gradient(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=0.01):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        average_returns = [0]*iterations
        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=learning_rate)

        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0
            prev_state = self.env.reset()

            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                probabilities = self.policy_network(prev_state)  # Forward pass

                assert probabilities.shape == (2,), f"{probabilities.shape}"
                assert 0.999 <= sum(probabilities) <= 1.001, f"{sum(probabilities)}"

                probable_actions = torch.distributions.Categorical(probabilities)
                action = probable_actions.sample()  # Choose 0 or 1

                assert action.item() in (0, 1), f"{action}"

                current_state, reward, done, _ = self.env.step(action.item())

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

    def do_reinforce_modified_policy_gradient_bias_subtracted(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=0.01, b=15.0):
        """
        DIKEO decide b ?
        b
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        average_returns = [0] * iterations
        optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=learning_rate)

        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            step = 0
            prev_state = self.env.reset()

            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                probabilities = self.policy_network(prev_state)  # Forward pass

                assert probabilities.shape == (2,), f"{probabilities.shape}"
                assert 0.999 <= sum(probabilities) <= 1.001, f"{sum(probabilities)}"

                probable_actions = torch.distributions.Categorical(probabilities)
                action = probable_actions.sample()  # Choose 0 or 1

                assert action.item() in (0, 1), f"{action}"

                current_state, reward, done, _ = self.env.step(action.item())

                current_trajectory["log_probabilities"].append(probable_actions.log_prob(action))
                current_trajectory["rewards"].append(reward)

                prev_state = current_state

                if done:  # episode terminated
                    T = len(current_trajectory["rewards"])  # Total steps
                    discounted_reward = sum((current_trajectory["rewards"][t] - b) * (gamma ** t) for t in
                                            range(len(current_trajectory["rewards"])))

                    objective_over_all_episodes = [current_trajectory["log_probabilities"][t] * sum(
                        [(gamma ** (t_bar - t)) * current_trajectory["rewards"][t_bar]])
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
    reinforce_types = {1: CartPole.do_reinforce,
                       2: CartPole.do_reinforce_modified_policy_gradient,
                       3: CartPole.do_reinforce_modified_policy_gradient_bias_subtracted}

    bot = CartPole(16)
    required_function = reinforce_types[2]
    required_function(bot, iterations=200, batch_size=500, learning_rate=0.01)



