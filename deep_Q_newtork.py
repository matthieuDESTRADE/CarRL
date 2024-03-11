import torch
from torch import nn
import numpy as np
import gymnasium as gym
from environment import CarEnv
from tqdm.notebook import tqdm
import itertools
from torchrl.data import ReplayBuffer


class EpsilonGreedy:
    def __init__(self,
                 epsilon_start: float,
                 epsilon_min: float,
                 epsilon_decay: float,
                 env: gym.Env,
                 q_network: nn.Module):

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: np.ndarray) -> np.int64:

        state = CarEnv.obs2tensor(state)
        if np.random.rand() > self.epsilon:
            action = self.q_network(state).argmax(dim=1).item()
        else:
            action = self.env.action_space.sample()

        return action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, last_epoch: int = -1, min_lr: float = 1e-6):

        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]


class DQNAgent():
    def __init__(self,
                 env: gym.Env,
                 q_network: torch.nn.Module,
                 target_q_network: torch.nn.Module,
                 target_q_network_sync_period: int,
                 device: torch.device,
                 gamma: float,
                 ):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.target_q_network_sync_period = target_q_network_sync_period
        self.device = device
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer()
        self.result_list = [[], [], []]
        self.train_index = 0

    def train(self,
              num_episodes: int,
              batch_size: int,
              loss_fn,
              epsilon_greedy: EpsilonGreedy,
              optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler,
              frame_skipping: int,
              ):
        iteration = 0
        self.train_index += 1

        for episode_index in tqdm(range(1, num_episodes)):
            state = self.env.reset()
            episode_reward = 0

            for t in itertools.count():
                # Get action, next_state and reward
                if t % frame_skipping == 0:
                    action = epsilon_greedy(state)

                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.add(
                    (state, action, reward, next_state, done))

                episode_reward += reward
                # Update the q_network weights with a batch of experiences from the buffer

                if len(self.replay_buffer) > batch_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(
                        batch_size)

                    # Convert to PyTorch tensors
                    batch_states_tensor = CarEnv.obs2tensor(
                        batch_states, self.device)
                    batch_actions_tensor = torch.tensor(
                        batch_actions, dtype=torch.long, device=self.device)
                    batch_rewards_tensor = torch.tensor(
                        batch_rewards, dtype=torch.float32, device=self.device)
                    batch_next_states_tensor = CarEnv.obs2tensor(
                        batch_next_states, self.device)
                    batch_dones_tensor = torch.tensor(
                        batch_dones, dtype=torch.float32, device=self.device)

                    loss = loss_fn(self.q_network(batch_states_tensor)[range(batch_actions_tensor.size(
                        0)), batch_actions_tensor], batch_rewards_tensor + self.gamma * torch.max(self.target_q_network(batch_next_states_tensor), axis=1).values * (1 - batch_dones_tensor))

                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    lr_scheduler.step()

                # Update the target q-network

                # Every few training steps (e.g., every 100 steps), the weights of the target network are updated with the weights of the Q-network

                if iteration % self.target_q_network_sync_period == 0:
                    self.target_q_network.load_state_dict(
                        self.q_network.state_dict())

                iteration += 1

                if done:
                    break

                state = next_state

            epsilon_greedy.decay_epsilon()

            self.result_list[0].append(episode_index)
            self.result_list[1].append(episode_reward)
            self.result_list[2].append(self.train_index)
            print(f'Episode {episode_index} - Reward: {episode_reward}')
