from pathlib import Path

import numpy as np

from agent.DQNAgent import DQNAgent
from environment.SimpleEnv import SimpleEnv
from environment.TargetArenaEnv import TargetArenaEnv

ROOT = Path(__file__).resolve().parents[1]


def test_simple_env_reset_and_goal():
    env = SimpleEnv(size=5)
    state = env.reset()
    assert state.shape == (2,)
    assert state.dtype == np.float32

    # Position juste à gauche de la cible (4, 0) -> action Droite (3)
    env._update_internal_state(np.array([3.0, 0.0], dtype=np.float32))
    next_state, reward, done = env.step(3)
    assert tuple(next_state) == (4.0, 0.0)
    assert reward == 10.0
    assert done is True


def test_target_arena_reset_state_and_bounds():
    env = TargetArenaEnv(size=10, render_mode=False)
    state = env.reset()

    assert state.shape == (14,)
    assert state.dtype == np.float32
    assert env.agent_pos != env.target_pos
    assert env.agent_pos not in env.blocked_cells
    assert env.target_pos not in env.blocked_cells


def test_target_arena_wall_obstacle_and_target_mechanics():
    env = TargetArenaEnv(size=10, render_mode=False)

    # 1. Collision contre le mur interne (3, 1) -> reste bloqué, pénalité -1.0
    env.agent_pos = (2, 1)
    env.target_pos = (9, 9)
    _, reward, done, info = env.step(3)  # Droite
    assert env.agent_pos == (2, 1)
    assert reward == -1.0
    assert done is False
    assert info["result"] == "wall"

    # 2. Collision contre l'obstacle mortel (1, 2) -> done=True, pénalité -10.0
    env.agent_pos = (0, 2)
    env.target_pos = (9, 9)
    _, reward, done, info = env.step(3)  # Droite
    assert env.agent_pos == (1, 2)
    assert reward == -10.0
    assert done is True
    assert info["result"] == "obstacle"

    # 3. Cible atteinte -> done=True, récompense +20.0
    env.agent_pos = (0, 0)
    env.target_pos = (1, 0)
    _, reward, done, info = env.step(3)  # Droite
    assert env.agent_pos == (1, 0)
    assert reward == 20.0
    assert done is True
    assert info["result"] == "target"


def test_pretrained_checkpoints_load_and_predict():
    grid_model = ROOT / "models" / "dqn_model.pth"
    arena_model = ROOT / "models" / "dqn_arena.pth"

    assert grid_model.exists()
    assert arena_model.exists()

    grid_agent = DQNAgent(state_shape=(2,), action_size=5)
    grid_agent.load(grid_model)
    assert 0 <= grid_agent.choose_action(np.array([0.0, 4.0], dtype=np.float32), epsilon=0.0) < 5

    arena_agent = DQNAgent(state_shape=(14,), action_size=4)
    arena_agent.load(arena_model)
    env = TargetArenaEnv(size=10, render_mode=False)
    assert 0 <= arena_agent.choose_action(env.reset(), epsilon=0.0) < 4