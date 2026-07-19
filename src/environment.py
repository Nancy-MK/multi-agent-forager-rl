"""
environment.py

A simple grid-world multi-agent foraging environment. Multiple agents move
around a grid collecting resources; a collision is recorded whenever two
agents occupy the same cell after a step, which is the primary safety
metric used in the independent-vs-shared policy comparison.
"""
import numpy as np

ACTIONS = ["up", "down", "left", "right", "stay"]
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}


class ForagingEnv:
    def __init__(self, grid_size=8, n_agents=4, n_resources=6, seed=None):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        self.rng.shuffle(cells)

        self.agent_positions = cells[: self.n_agents]
        self.resource_positions = set(cells[self.n_agents: self.n_agents + self.n_resources])
        self.collisions_this_episode = 0
        return self._get_state()

    def _get_state(self, agent_idx=None):
        """State = agent's own position + relative position of nearest resource."""
        if agent_idx is None:
            return [self._get_state(i) for i in range(self.n_agents)]

        pos = self.agent_positions[agent_idx]
        if self.resource_positions:
            nearest = min(
                self.resource_positions,
                key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]),
            )
            rel = (nearest[0] - pos[0], nearest[1] - pos[1])
        else:
            rel = (0, 0)
        return (pos, rel)

    def step(self, actions):
        """actions: list of action strings, one per agent."""
        new_positions = []
        for pos, action in zip(self.agent_positions, actions):
            dr, dc = ACTION_DELTAS[action]
            r = min(max(pos[0] + dr, 0), self.grid_size - 1)
            c = min(max(pos[1] + dc, 0), self.grid_size - 1)
            new_positions.append((r, c))

        # Collision = two or more agents end up on the same cell.
        collisions = len(new_positions) - len(set(new_positions))
        self.collisions_this_episode += collisions

        rewards = []
        for i, pos in enumerate(new_positions):
            reward = -0.01  # small step penalty encourages efficient foraging
            if pos in self.resource_positions:
                reward += 1.0
                self.resource_positions.discard(pos)
            if new_positions.count(pos) > 1:
                reward -= 0.5  # collision penalty
            rewards.append(reward)

        self.agent_positions = new_positions
        done = len(self.resource_positions) == 0
        return self._get_state(), rewards, done, {"collisions": collisions}
