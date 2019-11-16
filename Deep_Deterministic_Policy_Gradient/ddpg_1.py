import copy
import random

import numpy as np
import torch
import time
import torch.nn as nn

import gym
import pybullet
import pybulletgym.envs
import matplotlib.pyplot as plt

np.random.seed(1000)


# ReplayBuffer
class Replay:
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.env = env
        self.buffer_size = int(buffer_size)
        self.init_length = int(init_length)
        self.buffer = [0]*self.init_length
        self.collect_initial_samples(initial_sample_size=self.init_length)
        self.write_pointer = self.init_length - 1 # say 999

    def collect_initial_samples(self, initial_sample_size=1000):
        """
        Fills buffer with initial samples, each sample being a dictionary
        :return:
        """
        prev_state = self.env.reset()
        for sample in range(initial_sample_size):  # 1000 random samples
            action = self.env.action_space.sample()
            current_state, reward, done, _ = self.env.step(action)
            self.buffer[sample] = {"state": prev_state, "action": action, "reward": reward, "next_state": current_state}
            prev_state = current_state
            if done:  # If the bot is stuck at the terminal state, reset
                prev_state = self.env.reset()

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        assert isinstance(exp, dict) and len(exp) == 4 and tuple(exp.keys()) == ("state", "action","reward", "next_state"), f"{exp}"

        # Append to buffer until it's filled
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(exp)
        # Overwrite from beginning with new experiences
        else:
            self.write_pointer = (self.write_pointer + 1) % self.buffer_size
            self.buffer[self.write_pointer] = exp

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        sample = random.choices(population=self.buffer, k=N)
        assert len(sample) == N, f"{len(sample)}"
        return sample


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        Initialize the network

        Refer https://arxiv.org/pdf/1509.02971.pdf for architecture details!
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        if not isinstance(state, torch.Tensor):  # Convert numpy array to torch Tensor
            state = torch.from_numpy(state).float()  # Model parameters are in float, numpy by default goes with Double
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(action_dim + state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        if not isinstance(state, torch.Tensor):  # Convert numpy array to torch Tensor
            state = torch.from_numpy(state).float()  # Model parameters are in float, numpy by default goes with Double
        if not isinstance(action, torch.Tensor):  # Convert numpy array to torch Tensor
            action = torch.from_numpy(action).float()  # Model parameters are in float, numpy by default goes with Double

        input_x = torch.cat(tensors=(state, action), dim=0)
        x = torch.relu(self.fc1(input_x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPG():
    def __init__(
            self,
            env,
            action_dim,
            state_dim,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env

        # Create an actor and actor_target
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        # Make sure that both networks have the same initial weights

        # Create a critic and critic_target object
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        # Make sure that both networks have the same initial weights

        # Define the optimizer for the actor
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Define the optimizer for the critic
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Define a replay buffer
        self.ReplayBuffer = Replay(buffer_size=1e5,
                                   init_length=1e3,
                                   state_dim=state_dim,
                                   action_dim=action_dim,
                                   env=self.env)

    def save_all_models(self, path):
        """
        Save 4 models, actor, critic, actor target and critic targets
        :param path:
        :return:
        """
        models = self.actor, self.critic, self.actor_target, self.critic_target
        model_names = "actor", "critic", "actor_target", "critic_target"
        for model, model_name in zip(models, model_names):
            torch.save(model.state_dict(), path+"/"+model_name+"__.pt")

        print(f"Models saved at {path} directory")

    # A function to soft update target networks
    @staticmethod
    def weigh_sync(target_model, source_model, tau=0.001):
        """update target parameters with tau
        Using trick seen in https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
        and:
        https://discuss.pytorch.org/t/what-is-the-recommended-way-to-re-assign-update-values-in-a-variable-or-tensor/6125/2
        """
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau*source_param.data + (1-tau)*target_param.data)

    # Complete the function
    def update_target_networks(self):
        """
        A function to update the target networks
        """
        self.weigh_sync(self.actor_target, self.actor)
        self.weigh_sync(self.critic_target, self.critic)

    @staticmethod
    def update_network(optimizer, loss):
        """
        A function to update the function just once
        Requires only optimizer and the loss function to be passed
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, num_steps=100, evaluation_per_steps = 10):
        """
        Train the policy for the given number of iterations
        :param noise_type:
        :param num_steps:The number of steps to train the policy for
        """
        num_steps = int(num_steps)
        prev_state = self.env.reset()

        # Gaussian Noise parameters
        noise_mean = [0, 0]
        noise_cov = [[0.1, 0], [0, 0.1]]

        for step in range(num_steps):
            action = self.actor(prev_state)
            action = action.detach().numpy()

            additive_noise = np.random.multivariate_normal(noise_mean, noise_cov)
            action += additive_noise

            current_state, reward, done, _ = self.env.step(action)

            # Update Replay Buffer
            self.ReplayBuffer.buffer_add({"state": prev_state,
                                          "action": action,
                                          "reward": reward,
                                          "next_state": current_state})
            # Sampling a mini batch
            buffer_sample = self.ReplayBuffer.buffer_sample(N=self.batch_size)

            self.calculate_terms_for_losses(samples=buffer_sample)

            self.calculate_actor_loss()
            self.calculate_critic_loss()

            # Update the networks once!
            self.update_network(self.optimizer_actor, self.actor_loss)
            self.update_network(self.optimizer_critic, self.critic_loss)

            # Soft update target networks
            self.update_target_networks()

            if step % evaluation_per_steps == 0:
                total_return = self.evaluate_current_policy(self.actor)
                print(f"Iteration {step} : Return:{total_return}")

            prev_state = current_state
            if done:
                prev_state = self.env.reset()

    @staticmethod
    def evaluate_current_policy(current_policy):
        """
        Instantiate a new environment and collect rewards until return
        :param current_policy:
        :return:
        """

        eval_env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
        prev_state = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action = current_policy(prev_state)
            action = action.detach().numpy()
            current_state, reward, done, _ = eval_env.step(action)
            total_reward += reward
            prev_state = current_state

        return total_reward

    def calculate_critic_loss(self):
        """
        Calculate critic loss according to formula from DDPG paper
        :return:
        """
        self.critic_loss = (1/self.batch_size)*torch.sum((self.y_i-self.Q_s_i_a_i)**2)

    def calculate_actor_loss(self):
        """
        Calculate actor loss according to formula from DDPG paper
        :return:
        """
        self.actor_loss = -(1/self.batch_size)*torch.sum(self.Q_s_i_actor_a_i)  # Negative because of ascent

    def calculate_terms_for_losses(self, samples):
        """
        Follow DDPG Algorithm from paper for the exact formula to make the below expression clear!
        :param samples:
        :param sample:
        :return:
        """

        r_i = torch.zeros(self.batch_size)
        Gamma_i = torch.zeros_like(r_i)
        self.Q_s_i_a_i = torch.zeros_like(r_i)
        self.Q_s_i_actor_a_i = torch.zeros_like(r_i)

        for index, sample in enumerate(samples):
            state, action, reward, next_state = sample.values()
            r_i[index] = reward
            Gamma_i[index] = self.gamma * self.critic_target(next_state, self.actor_target(next_state))
            self.Q_s_i_a_i[index] = self.critic(state, action)
            self.Q_s_i_actor_a_i[index] = self.critic(state, self.actor(state))

        self.y_i = r_i + Gamma_i


if __name__ == "__main__":
    # Define the environment
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    SAVE_PATH = "/Users/ambareeshsnjayakumari/PycharmProjects/ReinforcementLearning/Deep_Deterministic_Policy_Gradient"

    ddpg_object = DDPG(
        env,
        action_dim=2,
        state_dim=8,
        critic_lr=1e-3,
        actor_lr=1e-3,
        gamma=0.99,
        batch_size=100,
    )

    # Train the policy
    # # DEBUG
    # ddpg_object.train(num_steps=100, evaluation_per_steps=10)
    ddpg_object.train(num_steps=2e5, evaluation_per_steps=100)

    ddpg_object.save_all_models(path=SAVE_PATH)

    # # Evaluate the final policy
    # state = env.reset()
    # done = False
    # while not done:
    #     action = ddpg_object.actor(state).detach().squeeze().numpy()
    #     next_state, r, done, _ = env.step(action)
    #     env.render()
    #     time.sleep(0.1)
    #     state = next_state
