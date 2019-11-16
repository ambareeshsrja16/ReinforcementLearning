import numpy as np
import gym
import pybulletgym.envs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
from torch.distributions import MultivariateNormal
import torch.optim


class PolicyNetwork(nn.Module):

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):  # Convert numpy array to torch Tensor
            state = torch.from_numpy(state).float()  # Model parameters are in float, numpy by default goes with Double
        mean = torch.tanh(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(state))))))
        return mean


class TwoLinkArm:

    def __init__(self):
        self.env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
        self.policy_network = PolicyNetwork()

    def do_reinforce_modified_policy_gradient_bias_subtracted(self, iterations=200, batch_size=1500, gamma=0.9,
                                                              learning_rate=3e-4, init_variance=0.01):
        """
        Function updates self.policy_network after performing self.iterations number of updates
        :return:
        """
        average_returns = [0] * iterations
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        variance = torch.nn.parameter.Parameter(torch.ones(2) * init_variance)  # for two dimensions

        b = 0  # Subtraction co-efficient
        std_dev_b = 1.0  # Division co-efficient

        for iteration in range(iterations):  # Updated once per loop
            net_reward = 0.0
            episodes = 0
            loss = 0.0
            step = 0
            prev_state = self.env.reset()

            G_tau = []
            current_trajectory = {"log_probabilities": [], "rewards": []}

            while step < batch_size:  # collection of data for batch size (say 500 steps)

                mean = self.policy_network(prev_state)  # Forward pass

                assert mean.shape == (2,), f"{mean.shape}"
                assert -2.00 <= sum(mean) <= 2.00, f"{sum(mean)}"

                covariance = torch.diag(abs(variance) + 1e-3 * torch.ones(2))
                # to avoid covariance going to zero! or negative
                probable_actions = MultivariateNormal(loc=mean, covariance_matrix=covariance)
                action = probable_actions.sample()  # Choose a 2-D vector for action

                current_state, reward, done, _ = self.env.step(action)

                current_trajectory["log_probabilities"].append(probable_actions.log_prob(action))
                current_trajectory["rewards"].append(reward)

                prev_state = current_state

                if done or step == batch_size - 1:  # episode terminated
                    T = len(current_trajectory["rewards"])  # Total steps
                    # For calculating b
                    for t in range(T):
                        discounted_reward = gamma ** (T - t) * current_trajectory["rewards"][t]
                        G_tau.append(discounted_reward)

                    discounted_reward = sum(current_trajectory["rewards"][t] * (gamma ** t) for t in
                                            range(len(current_trajectory["rewards"])))

                    objective_over_all_episodes = 0.0
                    for t in range(T):
                        summed_future_reward = 0.0
                        for t_bar in range(t, T):
                            summed_future_reward += (gamma ** (t_bar - t)) * (current_trajectory["rewards"][t_bar])
                        objective_over_all_episodes += current_trajectory["log_probabilities"][t] * ((summed_future_reward - b)/std_dev_b)

                    loss += objective_over_all_episodes
                    net_reward += discounted_reward
                    current_trajectory = {"log_probabilities": [],
                                          "rewards": []}  # Resetting for collecting next episode data
                    prev_state = self.env.reset()
                    episodes += 1

                step += 1

            # End of say, 500 steps, update policy
            optimizer.zero_grad()  # Clear gradients, to prevent accumulation
            loss = -1 * loss * (1 / episodes)
            # Objective function, -1 => Gradient Ascent (Torch does descent by default)
            loss.backward()  # Calculate gradient
            optimizer.step()  # Update neural net (aka policy)
            average_returns[iteration] = net_reward / episodes  # For plotting later

            assert len(G_tau) == batch_size, f"{batch_size, len(G_tau)}"
            b = sum(G_tau) / batch_size
            std_dev_b = np.std(G_tau)

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


if __name__ == "__main__":
    save_path = "/Users/ambareeshsnjayakumari/Desktop/Policy_Gradients"
    bot = TwoLinkArm()
    bot.do_reinforce_modified_policy_gradient_bias_subtracted(iterations=100,
                                                              batch_size=1000,
                                                              gamma=0.9,
                                                              learning_rate=3e-4,
                                                              init_variance=0.01)

    bot.env.render()
    steps = 0
    state = bot.env.reset()
    done = False
    import time
    # time.sleep(3)
    while steps < 300:
        a, _ = bot.policy_network(state)
        state, reward, done, info = bot.env.step(a)
        steps += 1
        bot.env.render()
        time.sleep(0.1)
    bot.env.close()



