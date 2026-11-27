import argparse
import random
import time
from pathlib import Path

import numpy as np
import pygame
import torch
from PIL import Image

from agent.DQNAgent import DQNAgent
from environment.TargetArenaEnv import TargetArenaEnv

ROOT = Path(__file__).resolve().parents[1]


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


def _capture_frame(screen: pygame.Surface) -> Image.Image:
    """Convertit la surface Pygame courante en image PIL."""
    raw = pygame.surfarray.array3d(screen)
    return Image.fromarray(np.transpose(raw, (1, 0, 2)))


def watch_agent(
    agent: DQNAgent,
    episodes: int = 15,
    max_steps: int = 80,
    fps: int = 8,
    record_gif: bool = False,
):
    env = TargetArenaEnv(size=10, render_mode=True, fps=fps)
    frames: list[Image.Image] = []

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            done = False
            steps = 0
            total_reward = 0.0
            status = "En cours"

            while not done and steps < max_steps and not env.quit_requested:
                env.render(f"Ep {ep}/{episodes} | Pas: {steps:2d} | Score: {total_reward:+5.1f}")
                if record_gif:
                    frames.append(_capture_frame(env.screen))

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
            if record_gif:
                # Maintient la frame finale ~0.6s pour rendre le succès lisible dans le GIF
                end_frame = _capture_frame(env.screen)
                frames.extend([end_frame] * max(1, fps // 2))
            else:
                time.sleep(0.5)
    finally:
        env.close()

    if record_gif and frames:
        gif_path = ROOT / "assets" / "demo_arena.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        print(f"GIF de démonstration enregistré dans : {gif_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Évaluation et démo Pygame de l'agent Double DQN.")
    parser.add_argument("--record", action="store_true", help="Enregistre une démo animée dans assets/demo_arena.gif")
    parser.add_argument("--episodes", type=int, default=None, help="Nombre d'épisodes visuels à afficher")
    parser.add_argument("--fps", type=int, default=8, help="Images par seconde")
    args = parser.parse_args()

    model_path = ROOT / "models" / "dqn_arena.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. Lance d'abord 'python -m scripts.train_arena'."
        )

    agent = DQNAgent(state_shape=(14,), action_size=4)
    agent.load(model_path)
    agent.q_network.eval()

    visual_episodes = args.episodes if args.episodes is not None else (6 if args.record else 15)

    with torch.no_grad():
        if args.record:
            random.seed(42)
            watch_agent(agent, episodes=visual_episodes, fps=args.fps, record_gif=True)
        else:
            benchmark(agent, episodes=200)
            watch_agent(agent, episodes=visual_episodes, fps=args.fps, record_gif=False)


if __name__ == "__main__":
    main()