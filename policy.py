"""Since the input space (~40*40*3*256) is large we will use CNN"""
from torch import nn
import torch


def recover_flattened(flat_params: torch.Tensor, model: torch.nn):
    """
    https://discuss.pytorch.org/t/how-to-flatten-and-then-unflatten-all-model-parameters/34730/2
    :param flat_params: [#params, 1]
    :param model: the model that gives the params with correct shapes
    """
    i = 0
    for param in model.parameters():
        n = param.numel()
        param.data = flat_params[i:i+n].view(param.shape)
        i += n


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


class SimpleQNetwork(nn.Module):
    """ A simple Q-network that takes in a 3x40x40 input and returns a 6x1 output. """

    def __init__(self, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*5*5, 6)

        self.model = nn.Sequential(
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.conv3, self.relu3, self.pool3,
            nn.Flatten(),
            self.fc1,
        )

    def forward(self, x):
        return self.model(x)


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
