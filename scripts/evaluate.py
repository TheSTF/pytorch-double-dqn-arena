import time
from pathlib import Path

import numpy as np
import torch

from agent.DQNAgent import DQNAgent
from environment.SimpleEnv import SimpleEnv

ACTION_NAMES = {
    0: "Haut ↑",
    1: "Bas ↓",
    2: "Gauche ←",
    3: "Droite →",
    4: "Inactif ·",
}

ACTION_ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
    4: "·",
}


def render_grid(size: int, agent_pos: tuple[int, int], target_pos: tuple[int, int]):
    """Affiche la grille en ASCII (y=0 en haut, x=0 à gauche)."""
    ax, ay = agent_pos
    tx, ty = target_pos

    border = "+" + "---+" * size
    print(border)
    for y in range(size):
        row = []
        for x in range(size):
            if (x, y) == (ax, ay) and (x, y) == (tx, ty):
                cell = "★"  # Agent sur la cible
            elif (x, y) == (ax, ay):
                cell = "A"
            elif (x, y) == (tx, ty):
                cell = "T"
            else:
                cell = "."
            row.append(f" {cell} ")
        print("|" + "|".join(row) + "|")
        print(border)


def print_policy_map(agent: DQNAgent, size: int, target_pos: tuple[int, int]):
    """Affiche l'action gloutonne (epsilon=0) choisie pour chaque case de la grille."""
    print("\n=== CARTE DE LA POLITIQUE APPRISE (epsilon = 0.0) ===")
    border = "+" + "---+" * size
    print(border)
    for y in range(size):
        row = []
        for x in range(size):
            if (x, y) == target_pos:
                row.append(" T ")
            else:
                state = np.array([x, y], dtype=np.float32)
                action = agent.choose_action(state, epsilon=0.0)
                row.append(f" {ACTION_ARROWS[action]} ")
        print("|" + "|".join(row) + "|")
        print(border)


def run_visual_episodes(
    env: SimpleEnv,
    agent: DQNAgent,
    episodes: int = 3,
    max_steps: int = 30,
    delay: float = 0.35,
):
    """Lance quelques épisodes avec affichage pas à pas dans le terminal."""
    target_pos = (env.size - 1, 0)

    print(f"\n=== OBSERVATION EN DIRECT ({episodes} épisodes) ===")
    for ep in range(1, episodes + 1):
        state = env.reset()
        start_pos = (int(state[0]), int(state[1]))
        optimal_steps = abs(target_pos[0] - start_pos[0]) + abs(target_pos[1] - start_pos[1])

        done = False
        steps = 0
        total_reward = 0.0

        print(f"\n--- Épisode {ep}/{episodes} | Départ : {start_pos} | Optimal théorique : {optimal_steps} pas ---")
        render_grid(env.size, start_pos, target_pos)
        time.sleep(delay)

        while not done and steps < max_steps:
            action = agent.choose_action(state, epsilon=0.0)
            next_state, reward, done = env.step(action)

            steps += 1
            total_reward += reward
            state = next_state

            pos = (int(state[0]), int(state[1]))
            print(f"Pas {steps:2d} | Action : {ACTION_NAMES[action]:<9} | Pos : {pos} | Reward : {reward:+.1f}")
            render_grid(env.size, pos, target_pos)
            time.sleep(delay)

        status = "✅ Cible atteinte" if done else "❌ Bloqué (boucle détectée)"
        print(f"Résultat : {status} en {steps} pas (récompense totale : {total_reward:+.1f})")


def run_benchmark(env: SimpleEnv, agent: DQNAgent, episodes: int = 100, max_steps: int = 30):
    """Évalue quantitativement l'agent sans hasard sur 100 épisodes."""
    target_pos = (env.size - 1, 0)
    successes = 0
    steps_list = []
    extra_steps_list = []
    rewards_list = []

    for _ in range(episodes):
        state = env.reset()
        start_pos = (int(state[0]), int(state[1]))
        # Si l'agent démarre sur la cible, il faut au moins 1 step() pour déclencher done=True
        optimal_steps = max(1, abs(target_pos[0] - start_pos[0]) + abs(target_pos[1] - start_pos[1]))

        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < max_steps:
            action = agent.choose_action(state, epsilon=0.0)
            state, reward, done = env.step(action)
            steps += 1
            total_reward += reward

        rewards_list.append(total_reward)
        if done:
            successes += 1
            steps_list.append(steps)
            extra_steps_list.append(steps - optimal_steps)

    print(f"\n=== BENCHMARK QUANTITATIF ({episodes} épisodes, epsilon = 0.0) ===")
    print(f"Taux de réussite        : {successes}/{episodes} ({successes / episodes * 100:.1f} %)")
    print(f"Récompense moyenne      : {np.mean(rewards_list):.2f}")
    if steps_list:
        print(f"Pas moyens (réussites)  : {np.mean(steps_list):.2f}")
        print(f"Écart moyen à l'optimal : +{np.mean(extra_steps_list):.2f} pas")


def main():
    model_path = Path(__file__).resolve().parents[1] / "models" / "dqn_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. Lance d'abord 'python -m scripts.train'."
        )

    env = SimpleEnv(size=5)
    agent = DQNAgent(state_shape=(2,), action_size=5)
    agent.load(model_path)
    agent.q_network.eval()

    target_pos = (env.size - 1, 0)
    print(f"Modèle chargé depuis : {model_path}")

    with torch.no_grad():
        print_policy_map(agent, env.size, target_pos)
        run_visual_episodes(env, agent, episodes=3, max_steps=30, delay=0.35)
        run_benchmark(env, agent, episodes=100, max_steps=30)


if __name__ == "__main__":
    main()