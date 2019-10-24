import numpy as np
import gym
import pybulletgym.envs


class FrozenLakeBot:
    def __init__(self, gamma=0.99, alpha=0.05, episodes=5000, annealing_function="linear"):

        assert isinstance(gamma, (float, int))

        self.env = gym.make("FrozenLake-v0")

        self.actions = self.env.nA  # 4
        self.states = self.env.nS  # 16

        # Action Value function or Q Function
        self.Q_s_a = np.zeros(shape=(self.states, self.actions))  # Arbitrary

        self.discount_rate = gamma  # Gamma for discounting future rewards
        self.learning_rate = alpha  # Learning rate for Q learning

        if annealing_function is "linear":
            self.annealing_function = lambda e: 1 - (e/self.episodes)
        elif annealing_function is "quadratic":
            self.annealing_function = lambda e: 1 - (e/self.episodes)**2
        elif annealing_function is "exponential":
            self.annealing_function = lambda e: np.exp(-e/self.episodes)

        self.episodes = episodes  # Max number of iterations allowed to achieve convergence

        self.success_rates = np.zeros(self.episodes//100)  # To story Success Rate every 100 episodes
        self.policy = np.zeros(self.states)  # To store optimal policy

        self.title = f"Gamma: {gamma}, Alpha: {alpha}, Annealing Function: {annealing_function}"
        self.save_filename = f"{annealing_function}_{gamma}_{alpha}.png"

    def TestPolicy(self, policy=lambda s: (s+1) % 4, render=False, allowed_steps=200):
        """
        Given deterministic policy return average rate of successful episodes over 100 trials
        :param render: True, renders state at each step
        :param allowed_steps: Terminates afer these many steps if the robot wanders around mindlessly
        :param policy: Given policy
        :return: average success rate (net_success_rate if trials is 100)
        """

        assert (type(policy) == np.ndarray and policy.shape == (16,)) \
               or hasattr(policy, "__call__"), "Function for np.array"

        if hasattr(policy, "__call__"):
            policy = np.fromfunction(policy, shape=(self.states,))

        net_success_rate = 0
        for episode in range(100):  # 100 trials
            state = self.env.reset()
            done = False
            steps = 0
            while not done:
                action = policy[state]  # deterministic
                steps += 1
                state, reward, done, info = self.env.step(action)
                if render:
                    self.env.render()
                done = done and (steps < allowed_steps)  # Terminate when Reach Goal/ Reach Hole / Ran out of steps

            net_success_rate += reward
            # If the latest reward=1, you reached the goal and loop was terminated, else reward 0

        #  Trials = 100, therefore net_success_rate is Average Success Rate %

        return net_success_rate

    def do_Q_Learning(self):
        """
        Q_learning TD control algorithm
        :return:
        """
        episode = 0
        while episode < self.episodes:
            current_state = int(self.env.reset())  # Initialize state
            done = False
            while not done:
                action = self.get_action_derived_from_Q_function(current_state, episode)
                next_state, reward, done, _ = self.env.step(action)
                self.update_Q_function(current_state, next_state, action, reward)
                current_state = next_state

            # Saving data
            if episode % 100 == 0:
                success_rate = self.TestPolicy(policy=self.get_policy_based_on_Q_function())
                try:
                    self.success_rates[episode//100] = success_rate
                except IndexError:
                    print("Ignoring this episode in plotting")
                    print(f"{episode, len(self.success_rates)}")
            # Logging
            if not episode % 1000:
                print(f"Ep:{episode} Average rate of success for the learned policy:", success_rate)
                # # DEBUG
                # print("Episode: ", episode)
                # print("\tAverage rate of success for the learned policy:", success_rate)

            episode += 1

        self.policy = self.get_policy_based_on_Q_function()

    def get_action_derived_from_Q_function(self, s, episode):
        """
        Get best action to be taken at current state
        :param s: current state
        :return: best action
        """
        assert isinstance(s, int) and s in range(self.states), f"{type(s), s}"
        assert isinstance(episode, int), f"{episode}"

        best_action = np.argmax(self.Q_s_a[s, :])
        random_action = np.random.choice(self.actions)

        # Epsilon greedy exploration-exploitation scheme
        # Linear annealing
        random_action_prob = self.annealing_function(episode)
        prob = [1 - random_action_prob, random_action_prob]

        action = np.random.choice([best_action, random_action], p=prob)
        assert action in range(4), f"{action}"

        return action

    # TODO Make this a property?
    def update_Q_function(self, s, next_state, action, reward):
        """
        Update self.Q_s_a based on action taken at s
        :param next_state:
        :param reward:
        :param s:
        :param action:
        :return:
        """
        assert isinstance(s, int) and isinstance(action, (int, np.int64)), f"State: {s, type(s)}, " \
                                                                           f"Action: {action, type(action)}"
        assert s in range(self.states), f"{s}"
        assert action in range(self.actions), f"{action}"
        assert isinstance(reward, float) and reward in (0,1), f"{reward}"

        max_action = np.argmax(self.Q_s_a[next_state, :])
        temp = self.learning_rate * \
               (reward + self.discount_rate*self.Q_s_a[next_state, max_action] - self.Q_s_a[s, action])
        self.Q_s_a[s, action] += temp

    def get_policy_based_on_Q_function(self):
        """
        Get policy (Derive from the current value function)
        :return:
        """
        # Make deterministic policy
        temp_policy = np.zeros(self.states)
        for s in range(self.states):
            temp_policy[s] = np.argmax(self.Q_s_a[s, :])
        return temp_policy

    def plot_success_rate(self, save_path=None, save=True, show =False):
        """
        Plot success rates present in self.success_rates
        :return:
        """
        if save_path:
            assert isinstance(save_path, str)

        import matplotlib.pyplot as plt
        import os

        plt.plot(self.success_rates)
        plt.title(self.title)
        plt.ylabel("Success Rate")
        plt.xlabel("Iterations | 1 unit = 100 Episodes")

        if save and save_path:
            plt.savefig(os.path.join(save_path, self.save_filename))
        else:
            plt.savefig(self.title, format='png')

        if show:
            plt.show()

        plt.close()


if __name__ == "__main__":

    save_path = "/Users/ambareeshsnjayakumari/Desktop/ECE276C/Tabular_Methods"
    function_dict = {1: "linear",
                     2: "quadratic",
                     3: "exponential"}

    bot = FrozenLakeBot(gamma=0.99, alpha=0.05, annealing_function=function_dict[2])
    bot.do_Q_Learning()
    bot.plot_success_rate(save_path)

    optimal_policy = bot.policy
    print("Final policy Success Rate", bot.TestPolicy(optimal_policy, render=False))








