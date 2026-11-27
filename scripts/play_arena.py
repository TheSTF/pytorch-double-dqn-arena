import time
from pathlib import Path

import torch

from agent.DQNAgent import DQNAgent
from environment.TargetArenaEnv import TargetArenaEnv


def benchmark(agent: DQNAgent, episodes: int = 200, max_steps: int = 100):
    env = TargetArenaEnv(size=10, render_mode=False)
    stats = {"target": 0, "obstacle": 0, "timeout": 0}

    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        result = "timeout"

        while not done and steps < max_steps:
            action = agent.choose_action(state, epsilon=0.0)
            state, _, done, info = env.step(action)
            steps += 1
            if done:
                result = info["result"]

        stats[result] += 1

    print(f"\n=== BENCHMARK ARÈNE ({episodes} épisodes, epsilon = 0.0) ===")
    print(f"Cibles atteintes : {stats['target']}/{episodes} ({stats['target'] / episodes * 100:.1f}%)")
    print(f"Collisions piège : {stats['obstacle']}/{episodes}")
    print(f"Bloqués (timeout): {stats['timeout']}/{episodes}\n")


def watch_agent(agent: DQNAgent, episodes: int = 15, max_steps: int = 80, fps: int = 8):
    env = TargetArenaEnv(size=10, render_mode=True, fps=fps)

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            done = False
            steps = 0
            total_reward = 0.0
            status = "En cours"

            while not done and steps < max_steps and not env.quit_requested:
                env.render(f"Ep {ep}/{episodes} | Pas: {steps:2d} | Score: {total_reward:+5.1f}")
                action = agent.choose_action(state, epsilon=0.0)
                state, reward, done, info = env.step(action)
                steps += 1
                total_reward += reward
                if done:
                    status = "CIBLE !" if info["result"] == "target" else "PIEGE !"

            if env.quit_requested:
                break

            if not done:
                status = "TIMEOUT"

            env.render(f"Ep {ep}/{episodes} | {status} | Pas: {steps} | Score: {total_reward:+5.1f}")
            time.sleep(0.5)
    finally:
        env.close()


def main():
    model_path = Path(__file__).resolve().parents[1] / "models" / "dqn_arena.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. Lance d'abord 'python -m scripts.train_arena'."
        )

    agent = DQNAgent(state_shape=(14,), action_size=4)
    agent.load(model_path)
    agent.q_network.eval()

    with torch.no_grad():
        benchmark(agent, episodes=200)
        watch_agent(agent, episodes=15, fps=8)


if __name__ == "__main__":
    main()