import numpy as np
import gym
import pybulletgym.envs


class FrozenLakeBot:
    def __init__(self):
        self.env = gym.make("FrozenLake-v0")

        self.actions = self.env.nA  # 4
        self.states = self.env.nS  # 16

        self.policy = np.zeros(self.states, dtype=int)  # Arbitrary initialization
        self.value_of_state = np.zeros_like(self.policy, dtype=float)

        self.discount_rate = 1  # Gamma for discounting future rewards

        self.iterations = 50  # Max number of iterations allowed to achieve convergence
        self.theta = 0.001

        self.transition_prob_matrix, self.reward_matrix = self.LearnModel()

        self.success_rates = []

        self.title = self.save_filename = "Value Iteration"

    def LearnModel(self):
        """
        Perform 10^5 random samples and obtain transition probabilities and rewards
        """
        joint_occurrence_matrix_sbar_a_s = np.zeros((self.states, self.actions, self.states))  # 16 states, 4 actions
        reward_matrix_sbar_a_s = np.zeros((self.states, self.actions, self.states))  # 16 states, 4 actions

        prev_state = self.env.reset()
        stuck = 0  # to prevent the robot being stuck in a terminal state forever
        for sample in range(10 ** 5):  # 10^5 random samples
            action = self.env.action_space.sample()
            current_state, reward, done, _ = self.env.step(action)

            joint_occurrence_matrix_sbar_a_s[current_state, action, prev_state] += 1
            reward_matrix_sbar_a_s[current_state, action, prev_state] += reward

            if prev_state == current_state:
                stuck += 1

            prev_state = current_state

            if stuck == 10:  # If the bot is stuck at the terminal step, reset
                stuck = 0
                prev_state = self.env.reset()

        assert np.sum(joint_occurrence_matrix_sbar_a_s) == 10 ** 5

        joint_prob_matrix_sbar_a_s = joint_occurrence_matrix_sbar_a_s / (10 ** 5)
        # P(a,b,c)

        joint_prob_matrix_a_s = np.sum(joint_prob_matrix_sbar_a_s, axis=0)
        # P(b,c)

        conditional_prob_matrix_sbar_given_a_s = joint_prob_matrix_sbar_a_s / joint_prob_matrix_a_s
        # P(a | b,c) = P(a,b,c)/ P(b,c)
        # Broadcasting
        # (16, 4, 16) / (4,16)
        # (16, 4, 16) / (1,4,16) Pad on left
        # (16,4, 16) / (16,4,16  Stretch

        reward_matrix_sbar_a_s = (reward_matrix_sbar_a_s > 0).astype(int)  # Values should be 1 or 0

        # # DEBUG
        # print("Reward matrix sum", reward_matrix_sbar_a_s.sum())

        # Checks
        for state in range(self.states):
            for action in range(self.actions):
                assert np.allclose(np.sum(conditional_prob_matrix_sbar_given_a_s[:, action, state]), 1), \
                    f"State: {state}, Action: {action} " \
                    f"Sum_p: {np.sum(conditional_prob_matrix_sbar_given_a_s[:, action, state])}"

        assert np.isclose(np.sum(conditional_prob_matrix_sbar_given_a_s), 64.0), \
            f"{np.sum(conditional_prob_matrix_sbar_given_a_s)}"

        return conditional_prob_matrix_sbar_given_a_s, reward_matrix_sbar_a_s

    def TestPolicy(self, policy=lambda s: (s + 1) % 4, render=False, allowed_steps=200):
        """
        Given deterministic policy return average rate of successful episodes over 100 trials
        :param render: True, renders state at each step
        :param allowed_steps: Terminates afer these many steps if the robot wanders around mindlessly
        :param policy: Given policy
        :return: average success rate (net_success_rate if trials is 100)
        """

        assert (type(policy) == np.ndarray and policy.shape == (16,)) or hasattr(policy, "__call__"), \
            "Function or np.array"

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

    def Value_Iteration(self):
        """
        Perform Value Iteration
        :return:
        """

        steps = 0
        while True:
            delta = 0
            for s in range(self.states):
                v = self.value_of_state[s]
                best_action = self.get_best_action(s)
                self.set_value_of_state(s, best_action)  # update V[s]
                delta = max(delta, abs(v - self.value_of_state[s]))
            steps += 1

            current_policy = self.get_policy_based_on_value_function()

            # # DEBUG
            # print(f"Average rate of success for the learned policy at {steps} :", self.TestPolicy(current_policy))

            success_rate = self.TestPolicy(current_policy)
            self.success_rates.append(success_rate)
            if not steps % 10:
                print("Average rate of success for the learned policy:", success_rate)

            # IF you want to iterate only for self.iterations regardless, regardless of convergence
            if steps == self.iterations:
                break

            # # IF you want to check for convergence
            # if delta < self.theta:
            #     break

        # Make deterministic optimal policy
        self.policy = self.get_policy_based_on_value_function()

    def get_best_action(self, s):
        """
        Get best action to be taken at current state
        :param s: current state
        :return: best action
        """
        assert isinstance(s, int) and s in range(self.states), f"{type(s), s}"

        best_action = None
        max_return = 0.0
        discounted_return_actions = np.zeros(4)
        for action in range(self.actions):
            discounted_return = 0.0
            for s_dash in range(self.states):
                discounted_return += self.get_discounted_return(s_dash, action, s)
            discounted_return_actions[action] = discounted_return

            if discounted_return == np.nan:
                raise AssertionError(f"State {s} Action {best_action}, All returns {discounted_return_actions}")
            if discounted_return >= max_return:
                max_return = discounted_return
                best_action = action

        assert isinstance(best_action, int), f"Best action at state {s} is {best_action}, " \
                                             f"All returns {discounted_return_actions}"
        return best_action

    # DIKEO make this a property?
    def set_value_of_state(self, s, action):
        """
        Update self.value_of_state based on action taken at s
        :param s:
        :param action:
        :return:
        """
        assert isinstance(s, int) and isinstance(action, (int, np.int64)), f"State: {s, type(s)}, " \
                                                                           f"Action: {action, type(action)}"
        assert s in range(self.states), f"{s}"
        assert action in range(self.actions), f"{action}"

        temp = 0.0
        for s_dash in range(self.states):
            temp += self.get_discounted_return(s_dash, action, s)  # update value of state 's'
        self.value_of_state[s] = temp

    def get_discounted_return(self, s_dash, action, s):
        """
        Calculate discounted return => Reward for taking action at step s to move to s_dash, and expected value from s_dash
        :param s_dash:
        :param action:
        :param s:
        :return: discounted_return
        """
        assert all(isinstance(i, (int, np.int64)) for i in (s_dash, action, s)), f"{s_dash},{action},{s}"
        assert s_dash in range(self.states) and s in range(self.states) and action in range(self.actions), \
            f"{s_dash},{action},{s}"

        discounted_return = self.transition_prob_matrix[s_dash, action, s] * (self.reward_matrix[s_dash, action, s] +
                                                                              self.discount_rate * self.value_of_state[
                                                                                  s_dash])
        assert type(discounted_return) == np.float64, f"{discounted_return, type(discounted_return)}"
        return discounted_return

    def get_policy_based_on_value_function(self):
        """
        Get policy (Derive from the current value function)
        :return:
        """
        # Make deterministic policy
        temp_policy = np.zeros_like(self.policy)
        for s in range(self.states):
            temp_policy[s] = self.get_best_action(s)

        return temp_policy

    def plot_success_rate(self, save_path=None, save=True, show=False):
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
        plt.xlabel("Iterations")

        if save and save_path:
            plt.savefig(os.path.join(save_path, self.save_filename))
        else:
            plt.savefig(self.title, format='png')

        if show:
            plt.show()

        plt.close()


if __name__ == "__main__":
    save_path = "/Users/ambareeshsnjayakumari/Desktop/Tabular_Methods"

    bot = FrozenLakeBot()
    bot.Value_Iteration()
    bot.plot_success_rate(save_path=save_path)
    optimal_policy = bot.policy
    print("Final policy Success Rate", bot.TestPolicy(policy=optimal_policy, render=False))











