"""
agents.py

Two Q-learning agent architectures compared in this study:

- IndependentQAgents: every agent keeps and updates its own Q-table, with no
  information shared between agents.
- SharedQAgents: all agents read from and update a single, shared Q-table,
  representing policy coordination.

Both use epsilon-greedy exploration with a configurable annealing schedule,
since the annealing rate itself is one of the hyperparameters studied for
its effect on behavioural unpredictability.
"""
from collections import defaultdict

import numpy as np

from environment import ACTIONS


def _state_key(state):
    pos, rel = state
    return (pos, rel)


class _QTable:
    def __init__(self, n_actions, lr=0.1, gamma=0.95):
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma

    def best_action_idx(self, state):
        return int(np.argmax(self.q[_state_key(state)]))

    def update(self, state, action_idx, reward, next_state):
        key, next_key = _state_key(state), _state_key(next_state)
        best_next = np.max(self.q[next_key])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[key][action_idx]
        self.q[key][action_idx] += self.lr * td_error


class _EpsilonSchedule:
    def __init__(self, start=1.0, end=0.05, decay_episodes=800):
        self.start = start
        self.end = end
        self.decay_episodes = decay_episodes

    def value(self, episode):
        frac = min(episode / self.decay_episodes, 1.0)
        return self.start + frac * (self.end - self.start)


class IndependentQAgents:
    """Each agent has its own Q-table; no information is shared."""

    def __init__(self, n_agents, lr=0.1, gamma=0.95, epsilon_schedule=None, seed=None):
        self.n_agents = n_agents
        self.tables = [_QTable(len(ACTIONS), lr, gamma) for _ in range(n_agents)]
        self.epsilon_schedule = epsilon_schedule or _EpsilonSchedule()
        self.rng = np.random.default_rng(seed)

    def act(self, states, episode):
        eps = self.epsilon_schedule.value(episode)
        actions = []
        for i, state in enumerate(states):
            if self.rng.random() < eps:
                idx = self.rng.integers(len(ACTIONS))
            else:
                idx = self.tables[i].best_action_idx(state)
            actions.append(idx)
        return actions

    def update(self, states, actions, rewards, next_states):
        for i in range(self.n_agents):
            self.tables[i].update(states[i], actions[i], rewards[i], next_states[i])


class SharedQAgents:
    """All agents read from and update a single shared Q-table."""

    def __init__(self, n_agents, lr=0.1, gamma=0.95, epsilon_schedule=None, seed=None):
        self.n_agents = n_agents
        self.table = _QTable(len(ACTIONS), lr, gamma)
        self.epsilon_schedule = epsilon_schedule or _EpsilonSchedule()
        self.rng = np.random.default_rng(seed)

    def act(self, states, episode):
        eps = self.epsilon_schedule.value(episode)
        actions = []
        for state in states:
            if self.rng.random() < eps:
                idx = self.rng.integers(len(ACTIONS))
            else:
                idx = self.table.best_action_idx(state)
            actions.append(idx)
        return actions

    def update(self, states, actions, rewards, next_states):
        for i in range(self.n_agents):
            self.table.update(states[i], actions[i], rewards[i], next_states[i])
