import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.DQNAgent import DQNAgent
from environment.TargetArenaEnv import TargetArenaEnv

# --- Hyperparamètres optimisés (Arène 10x10 + Double DQN) ---
EPISODES = 4000
BATCH_SIZE = 128
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
DECAY_RATE = 0.9988      # ~0.05 vers l'épisode 2500, ~0.01 vers l'épisode 3800
MAX_STEPS_PER_EPISODE = 100
TARGET_UPDATE_FREQ = 500
LEARN_EVERY = 2
MEMORY_CAPACITY = 50000
HIDDEN_DIM = 128

EVAL_EVERY = 200
EVAL_EPISODES = 100


def evaluate_greedy(agent: DQNAgent, episodes: int = EVAL_EPISODES) -> tuple[float, int, int]:
    """Évalue la politique déterministe (epsilon = 0.0) sur un environnement dédié."""
    eval_env = TargetArenaEnv(size=10, render_mode=False)
    targets = 0
    obstacles = 0
    timeouts = 0

    agent.q_network.eval()
    try:
        with torch.no_grad():
            for _ in range(episodes):
                state = eval_env.reset()
                done = False
                steps = 0
                result = "timeout"

                while not done and steps < MAX_STEPS_PER_EPISODE:
                    action = agent.choose_action(state, epsilon=0.0)
                    state, _, done, info = eval_env.step(action)
                    steps += 1
                    if done:
                        result = info["result"]

                if result == "target":
                    targets += 1
                elif result == "obstacle":
                    obstacles += 1
                else:
                    timeouts += 1
    finally:
        agent.q_network.train()

    success_rate = targets / episodes * 100.0
    return success_rate, obstacles, timeouts


def train_arena():
    env = TargetArenaEnv(size=10, render_mode=False)
    agent = DQNAgent(
        state_shape=(14,),
        action_size=4,
        lr=1e-3,
        memory_capacity=MEMORY_CAPACITY,
        target_update_freq=TARGET_UPDATE_FREQ,
        hidden_dim=HIDDEN_DIM,
    )

    model_path = Path(__file__).resolve().parents[1] / "models" / "dqn_arena.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    epsilon = INITIAL_EPSILON
    total_rewards = []
    success_history = []
    recent_successes = deque(maxlen=50)

    eval_episodes = []
    eval_success_rates = []
    best_greedy_rate = -1.0
    global_step = 0

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        succeeded = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)

            step_count += 1
            global_step += 1

            agent.remember(state, action, reward, next_state, done)
            if global_step % LEARN_EVERY == 0:
                agent.learn(BATCH_SIZE, GAMMA)

            state = next_state
            episode_reward += reward
            if info["result"] == "target":
                succeeded = 1

        epsilon = max(FINAL_EPSILON, epsilon * DECAY_RATE)
        total_rewards.append(episode_reward)
        success_history.append(succeeded)
        recent_successes.append(succeeded)

        if episode % 50 == 0:
            rolling_rate = sum(recent_successes) / len(recent_successes) * 100.0
            avg_reward = float(np.mean(total_rewards[-50:]))
            print(
                f"Épisode {episode:4d}/{EPISODES} | "
                f"Train (50 ép.) : {rolling_rate:5.1f}% | "
                f"Récompense moy. : {avg_reward:+6.2f} | "
                f"Epsilon : {epsilon:.3f}"
            )

        if episode % EVAL_EVERY == 0:
            greedy_rate, eval_obs, eval_to = evaluate_greedy(agent, episodes=EVAL_EPISODES)
            eval_episodes.append(episode)
            eval_success_rates.append(greedy_rate)

            saved_tag = ""
            if greedy_rate >= best_greedy_rate:
                best_greedy_rate = greedy_rate
                agent.save(model_path)
                saved_tag = " [SAUVEGARDÉ]"

            print(
                f"  --> Éval sans hasard ({EVAL_EPISODES} ép., ε=0) : "
                f"{greedy_rate:5.1f}% | Pièges: {eval_obs} | Timeouts: {eval_to}{saved_tag}"
            )

    print(f"\nMeilleur taux de réussite sans hasard (ε=0) : {best_greedy_rate:.1f}%")
    print(f"Meilleur modèle sauvegardé dans : {model_path}")
    return total_rewards, success_history, eval_episodes, eval_success_rates


def plot_history(rewards, successes, eval_episodes, eval_rates):
    output_path = Path(__file__).resolve().parents[1] / "assets" / "training_history_arena.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window = 50
    episodes_axis = range(1, len(rewards) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    # Graphique 1 : Récompenses
    ax1.plot(episodes_axis, rewards, alpha=0.22, label="Récompense par épisode")
    if len(rewards) >= window:
        avg_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window, len(rewards) + 1),
            avg_r,
            linewidth=2,
            label=f"Moyenne ({window} ép.)",
        )
    ax1.set_ylabel("Récompense")
    ax1.set_title("Entraînement Double DQN - Arène 2D (Murs & Obstacles)")
    ax1.grid(True)
    ax1.legend()

    # Graphique 2 : Taux de réussite (entraînement vs évaluation sans hasard)
    if len(successes) >= window:
        avg_s = np.convolve(successes, np.ones(window) / window, mode="valid") * 100.0
        ax2.plot(
            range(window, len(successes) + 1),
            avg_s,
            color="green",
            alpha=0.8,
            label=f"Entraînement avec ε ({window} ép.)",
        )

    if eval_episodes:
        ax2.plot(
            eval_episodes,
            eval_rates,
            color="crimson",
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"Évaluation sans hasard (ε = 0.0, {EVAL_EPISODES} ép.)",
        )

    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Réussite (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Graphique enregistré dans : {output_path}")
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    rewards, successes, eval_episodes, eval_rates = train_arena()
    plot_history(rewards, successes, eval_episodes, eval_rates)