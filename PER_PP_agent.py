import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4/8.0               # learning rate
UPDATE_EVERY = 4        # how often to update the network

ALPHA = 0.6
BETA_START = 0.4              # technically
EPS_EDGE = .0001              # avoids division by 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

size_vector = np.arange(64)


class PER_PP_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, lr, beta, alpha, state_size, action_size, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.episode_num = 0
        self.beta = beta
        self.alpha = alpha

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # uncomment if you want to linearly anneal beta
        # if done:
        #     self.episode_num += 1
        #     self.beta = min(self.episode_num *
        #                     (1.0-BETA_START) / 500.0 + BETA_START, 1.0)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # experiences = self.memory.sample()
                experiences = self.memory.alt_sample(self.beta, self.alpha)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, p, w) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, _, weights = experiences

        argmax = self.qnetwork_local(
            next_states).detach().max(1)[1]

        Q_targets_next = self.qnetwork_target(
            next_states).detach()[size_vector, argmax].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute temporal difference error
        TD_errors = Q_targets - \
            self.qnetwork_local(states).detach().gather(1, actions)

        # update transition priorities
        self.memory.update_priorities(TD_errors)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(weights * Q_expected, weights * Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        if not self.memory:
            e = self.experience(state, action, reward, next_state, done, 1.0)
        else:
            max_priority = max([e.priority for e in self.memory])
            e = self.experience(state, action, reward,
                                next_state, done, max_priority)

        self.memory.append(e)

    def update_priorities(self, td_errors):
        """Update transition priorites in memory."""
        for sample, td_error in zip(self.samples, td_errors):
            experience = self.memory[sample]
            e = self.experience(experience.state, experience.action, experience.reward,
                                experience.next_state, experience.done, abs(td_error.item())+EPS_EDGE)
            self.memory[sample] = e

    def alt_sample(self, beta, alpha):
        """Randomly sample a batch of experiences from memory."""
        priorities = np.array(
            [e.priority for e in self.memory if e is not None])
        total_priority = np.sum(priorities**alpha)
        probabilities = priorities ** alpha / total_priority
        self.samples = np.random.choice(
            np.arange(len(self.memory)), size=self.batch_size, replace=False, p=probabilities)  # get sample indexes
        N = len(probabilities)  # replay buffer size
        # max importance-sampling (IS) weights
        max_weight = np.max((probabilities * N) ** -beta)
        # max importance-sampling (IS) weights
        # max_weight = np.max((probabilities[self.samples] * N) ** -beta)  # alternate test

        weights = torch.from_numpy(
            np.vstack(((probabilities[self.samples] * N) ** -beta)/max_weight)).float().to(device)  # importance-sampling (IS) weights

        states = torch.from_numpy(
            np.vstack([self.memory[e].state for e in self.samples])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([self.memory[e].action for e in self.samples])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([self.memory[e].reward for e in self.samples])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([self.memory[e].next_state for e in self.samples])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([self.memory[e].done for e in self.samples]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(
            np.vstack([self.memory[e].priority for e in self.samples])).float().to(device)

        return (states, actions, rewards, next_states, dones, priorities, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
