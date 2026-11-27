import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Hyperparamètres par défaut ---
LEARNING_RATE = 1e-3
MEMORY_CAPACITY = 10000
TARGET_UPDATE_FREQ = 100
HIDDEN_DIM = 128


class QNetwork(nn.Module):
    """Réseau neuronal pour estimer les valeurs Q(s, a)."""

    def __init__(self, state_size: tuple, action_size: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        state_dim = int(np.prod(state_size))

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
        elif isinstance(state, torch.Tensor):
            state_tensor = state
        else:
            raise TypeError("Le type d'état passé au réseau n'est pas supporté.")

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        x = torch.relu(self.fc1(state_tensor))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Agent Double DQN avec Replay Memory et Target Network."""

    def __init__(
        self,
        state_shape: tuple,
        action_size: int,
        lr: float = LEARNING_RATE,
        memory_capacity: int = MEMORY_CAPACITY,
        target_update_freq: int = TARGET_UPDATE_FREQ,
        hidden_dim: int = HIDDEN_DIM,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # 1. Réseau principal (Policy Network)
        self.q_network = QNetwork(state_shape, action_size, hidden_dim=hidden_dim)

        # 2. Réseau cible (Target Network)
        self.target_network = QNetwork(state_shape, action_size, hidden_dim=hidden_dim)
        self.update_target_network()
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_capacity)

    def update_target_network(self):
        """Copie les poids du réseau principal vers le Target Network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Stocke une transition dans la mémoire de rejeu."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray, epsilon: float) -> int:
        """Stratégie epsilon-greedy."""
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def learn(self, batch_size: int, gamma: float = 0.99):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        # Valeurs Q actuelles prédites par le Policy Network
        current_q = self.q_network(states).gather(1, actions).squeeze(1)

        # Double DQN : sélection par q_network, évaluation par target_network
        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + gamma * next_q * (1.0 - dones)

        # Perte de Huber + gradient clipping
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Synchronisation périodique du Target Network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath):
        state_dict = torch.load(filepath, weights_only=True)
        self.q_network.load_state_dict(state_dict)
        self.update_target_network()

    def __len__(self):
        return len(self.memory)