import torch
from torch import nn
import numpy as np
import gymnasium as gym
from environment import CarEnv
from tqdm.notebook import tqdm
import itertools
from torchrl.data import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 32)
        self.relu3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, x):
        # Define the forward pass of the CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class OtherQNetwork(nn.Module):
    """
    Expects a 3x40x40 input and returns a 6x1 output.

    """

    def __init__(self, action_dim):
        super().__init__()
        # Layer 1: Convolutional Layer, Pooling, ReLU
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        # Layer 2: Convolutional Layer, Pooling, ReLU
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        # Layer 3: Convolutional Layer, ReLU
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=3, padding=1)
        self.relu3 = nn.ReLU()

        # Layer 4: Fully Connected Layer, ReLU
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.relu4 = nn.ReLU()

        # layer 5: LSTM
        self.lstm = nn.LSTM(256, action_dim, batch_first=True)

        self.model = nn.Sequential(
            self.conv1, self.pool1, self.relu1,
            self.conv2, self.pool2, self.relu2,
            self.conv3, self.relu3,
            nn.Flatten(),
            self.fc1, self.relu4,
            self.lstm,
        )

    def forward(self, x):
        # Define the forward pass of the CNN
        output, (hidden_state, cell_state) = self.model(x)
        probas = nn.functional.softmax(output, dim=1)
        return probas


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
            rdm = -1
        else:
            action = self.env.action_space.sample()
            rdm = 1

        return action, rdm

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
                 frame_skipping: int,
                 ):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.target_q_network_sync_period = target_q_network_sync_period
        self.device = device
        self.gamma = gamma
        self.frame_skipping = frame_skipping
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
              ):
        iteration = 0
        self.train_index +=1

        for episode_index in tqdm(range(1, num_episodes)):
            state = self.env.reset()
            episode_reward = 0
            rdm = 10

            for t in itertools.count():

                # Get action, next_state and reward

                if np.abs(rdm) >= self.frame_skipping:
                    action, rdm = epsilon_greedy(state)
                elif rdm < 0:
                    rdm -= 1
                elif rdm >= 1:
                    rdm += 1

                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.add((state, action, reward, next_state, done))

                episode_reward += reward
                # Update the q_network weights with a batch of experiences from the buffer

                if len(self.replay_buffer) > batch_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(
                        batch_size)

                    # Convert to PyTorch tensors
                    batch_states_tensor = CarEnv.obs2tensor(batch_states, self.device)
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