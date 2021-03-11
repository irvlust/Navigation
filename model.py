import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QDuelingNetwork(nn.Module):
    """Dueling Architecture Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QDuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc_value1 = nn.Linear(fc1_units, fc2_units)
        self.fc_value2 = nn.Linear(fc2_units, 1)
        self.fc_adv1 = nn.Linear(fc1_units, fc2_units)
        self.fc_adv2 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        stateval = F.relu(self.fc_value1(x))
        actadv = F.relu(self.fc_adv1(x))
        stateval = F.relu(self.fc_value2(stateval))
        actadv = F.relu(self.fc_adv2(actadv))
        average = torch.mean(actadv, dim=1, keepdim=True)
        Qout = stateval + actadv - average
        return Qout
