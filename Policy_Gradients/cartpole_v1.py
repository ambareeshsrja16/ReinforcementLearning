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

    def __init__(self):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):  # Convert numpy array to torch Tensor
            state = torch.from_numpy(state).float()  # Model parameters are in float, numpy by default goes with Double

        probabilities = torch.softmax(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(state))))), dim=0)
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
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        for iteration in range(iterations):  # Update policy once per iteration
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            step = 0
            prev_state = self.env.reset()

            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                probabilities = self.policy_network(prev_state)  # Forward pass

                assert probabilities.shape == (2,), f"{probabilities.shape}"
                assert 0.999 <= sum(probabilities) <= 1.001, f"{sum(probabilities)}"

                probable_actions = torch.distributions.Categorical(probabilities)
                action = probable_actions.sample()  # Choose 0 or 1

                # # Debug
                # print(action, action.item())
                assert action.item() in (0, 1), f"{action}"

                current_state, reward, done, _ = self.env.step(action.item())

                current_trajectory["log_probabilities"].append(probable_actions.log_prob(action))
                current_trajectory["rewards"].append(reward)

                # # DEBUG
                # print(f"Reward for step {step} = {reward}")
                # print(f"Log prob : {probable_actions.log_prob(action)}")

                prev_state = current_state

                if done or step == batch_size-1:  # episode terminated or 500 steps

                    discounted_reward = sum(current_trajectory["rewards"][t] * (gamma**t) for t in range(len(current_trajectory["rewards"])))
                    # calculate discounted reward for trajectory
                    sum_of_log_prob = sum(current_trajectory["log_probabilities"])

                    # # DEBUG
                    # print(f"SUM of Log prob for episode {episodes} = {sum_of_log_prob}")
                    # print(f"DISC Reward  : {discounted_reward}")

                    loss += discounted_reward * sum_of_log_prob

                    net_reward += discounted_reward
                    # DIKEO CHECK Which?
                    # net_reward += sum(current_trajectory["rewards"])

                    # # DEBUG
                    # print(f"Discounted Reward for iteration: {iteration} = {discounted_reward}")
                    # print(f"Net loss : {loss}")
                    # print(f"Net reward : {net_reward}")

                    # Resetting for collecting next episode data
                    current_trajectory = {"log_probabilities": [], "rewards": []}
                    prev_state = self.env.reset()
                    episodes += 1

                step += 1

            # End of 500 steps, update policy
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            loss = -1 * loss * (1/episodes)  # Objective function, -1 => Gradient Ascent (Torch does descent by default)
            loss.backward()  # Calculate gradient
            optimizer.step()  # Update neural net (aka policy)

            average_returns[iteration] = net_reward/episodes  # For plotting later

            # # DEBUG
            print(f"Discounted Reward for iteration: {iteration} = {discounted_reward}")
            print(f"Avg loss : {loss}")
            print(f"Avg reward : {average_returns[iteration]}")

            print(f"Iteration: {iteration}, "
                  f"Average Return : {average_returns[iteration]}, "
                  f"Episodes: {episodes}")

        self.plot_average_returns(average_returns)

    def do_reinforce_modified_policy_gradient(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=3e-4):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        average_returns = [0]*iterations
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
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

                if done or step == batch_size-1:  # episode terminated
                    T = len(current_trajectory["rewards"])  # Total steps in trajectory

                    discounted_reward = sum(current_trajectory["rewards"][t] * (gamma**t) for t in range(len(current_trajectory["rewards"])))

                    objective_over_all_episodes = [current_trajectory["log_probabilities"][t] * sum([(gamma**(t_bar-t)) * current_trajectory["rewards"][t_bar]])
                                                    for t in range(T)
                                                    for t_bar in range(t, T)]

                    objective_over_all_episodes = sum(objective_over_all_episodes)

                    # # OR
                    # objective_over_all_episodes = 0.0
                    # for t in range(T):
                    #     summed_future_reward = 0.0
                    #     for t_bar in range(t, T):
                    #         summed_future_reward += (gamma**(t_bar-t)) * current_trajectory["rewards"][t_bar]
                    #     objective_over_all_episodes += current_trajectory["log_probabilities"][t] * summed_future_reward
                    # #

                    loss += objective_over_all_episodes
                    net_reward += discounted_reward
                    current_trajectory = {"log_probabilities": [], "rewards": []}  # Resetting for collecting next episode data
                    prev_state = self.env.reset()
                    episodes += 1

                step += 1

            # End of 500 steps, time to update policy

            average_returns[iteration] = net_reward/episodes  # For plotting later

            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            loss = -1 * loss * (1/episodes)  # Objective function, -1 => Gradient Ascent (Torch does descent by default)
            loss.backward()  # Calculate gradient
            optimizer.step()  # Update neural net (aka policy)

            # # DEBUG

            print(f"Iteration: {iteration}, "
                  f"Average Return : {average_returns[iteration]}, "
                  f"Episodes: {episodes}")
            print(f"Avg loss : {loss}")


        self.plot_average_returns(average_returns)

    def do_reinforce_modified_policy_gradient_bias_subtracted(self, iterations=200, batch_size=500, gamma=0.99, learning_rate=3e-4):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        average_returns = [0] * iterations
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        b = 0  # Subtraction co-efficient
        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            step = 0
            prev_state = self.env.reset()

            G_tau = []

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

                if done or step == batch_size-1:  # episode terminated
                    T = len(current_trajectory["rewards"])  # Total steps
                    # For calculating b
                    for t in range(T):
                        discounted_reward = gamma**(T-t) * current_trajectory["rewards"][t]
                        G_tau.append(discounted_reward)

                    discounted_reward = sum(current_trajectory["rewards"][t] * (gamma ** t) for t in
                                            range(len(current_trajectory["rewards"])))

                    # objective_over_all_episodes = [current_trajectory["log_probabilities"][t] * sum([(gamma ** (t_bar - t)) * (current_trajectory["rewards"][t_bar]-b)])
                    #                                for t in range(T)
                    #                                for t_bar in range(t, T)]
                    #
                    # objective_over_all_episodes = sum(objective_over_all_episodes)

                    # # # OR
                    objective_over_all_episodes = 0.0
                    for t in range(T):
                        summed_future_reward = 0.0
                        for t_bar in range(t, T):
                            summed_future_reward += (gamma**(t_bar-t)) * (current_trajectory["rewards"][t_bar] - b)
                        objective_over_all_episodes += current_trajectory["log_probabilities"][t] * summed_future_reward
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
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            loss = -1 * loss * (1/episodes)  # Objective function, -1 => Gradient Ascent (Torch does descent by default)
            loss.backward()  # Calculate gradient
            optimizer.step()  # Update neural net (aka policy)
            average_returns[iteration] = net_reward / episodes  # For plotting later

            assert len(G_tau) == batch_size, f"{batch_size, len(G_tau)}"
            b = sum(G_tau)/batch_size

            # # DEBUG
            print(f"Iteration: {iteration}, "
                  f"Average Return : {average_returns[iteration]}, "
                  f"Episodes: {episodes}")
            print(f"Avg loss : {loss}")

        self.plot_average_returns(average_returns, title=f"Average Returns | Batch Size: {batch_size}")

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

    def test_policy(self):
        """
        Test the given policy
        :return:
        """
        self.env.render("")
        prev_state = self.env.reset()
        done = False
        steps = 0
        # while not done:
        while steps < 100:
            probabilities = self.policy_network(prev_state)  # Forward pass
            assert probabilities.shape == (2,), f"{probabilities.shape}"
            assert 0.999 <= sum(probabilities) <= 1.001, f"{sum(probabilities)}"

            probable_actions = torch.distributions.Categorical(probabilities)
            action = probable_actions.sample()  # Choose 0 or 1

            assert action.item() in (0, 1), f"{action}"

            current_state, reward, done, _ = self.env.step(action.item())
            self.env.render()
            prev_state = current_state
            steps += 1
            if done:
                print("Complete", steps)

        self.env.close()


if __name__ == "__main__":
    save_path = "/Users/ambareeshsnjayakumari/Desktop/Policy_Gradients"
    reinforce_algo = {1: CartPole.do_reinforce,
                      2: CartPole.do_reinforce_modified_policy_gradient,
                      3: CartPole.do_reinforce_modified_policy_gradient_bias_subtracted}

    bot = CartPole()
    required_function = reinforce_algo[3]
    required_function(bot, iterations=200, batch_size=1000, learning_rate=3e-4)

    bot.test_policy()




