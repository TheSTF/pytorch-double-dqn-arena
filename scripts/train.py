import torch
import numpy as np
import random
from environment.SimpleEnv import SimpleEnv
from agent.DQNAgent import DQNAgent
import matplotlib.pyplot as plt
from pathlib import Path

# --- Hyperparamètres d'Entraînement (À ajuster) ---
EPISODES = 1000        # Nombre total d'épisodes à simuler
BATCH_SIZE = 64        # Taille du mini-batch pour l'apprentissage
GAMMA = 0.99           # Facteur d'actualisation
INITIAL_EPSILON = 1.0  # Commence par explorer au maximum
FINAL_EPSILON = 0.05   # Niveau minimum d'exploration
DECAY_RATE = 0.99     # Taux de décroissance de epsilon
MAX_STEPS_PER_EPISODE = 200
TARGET_UPDATE_FREQ = 100
LEARN_EVERY = 4  # 1 mise à jour tous les 4 pas

def train_agent():
    """Orchestre le cycle complet d'entraînement DQN."""
    print("=============================================")
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT DQN")
    print("=============================================")

    # 1. Initialisation de l'Environnement et de l'Agent
    env = SimpleEnv(size=5)
    agent = DQNAgent(
        state_shape=(2,),
        action_size=5,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    epsilon = INITIAL_EPSILON
    total_rewards = []
    global_step = 0
    
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_losses = []

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)

            step_count += 1
            global_step += 1

            # On stocke uniquement le vrai état terminal `done` (pas la troncature)
            agent.remember(state, action, reward, next_state, done)

            # Apprentissage régulier pendant l'épisode
            if global_step % LEARN_EVERY == 0:
                loss = agent.learn(BATCH_SIZE, GAMMA)
                if loss is not None:
                    episode_losses.append(loss)

            state = next_state
            episode_reward += reward

        epsilon = max(FINAL_EPSILON, epsilon * DECAY_RATE)
        total_rewards.append(episode_reward)

        if episode % 10 == 0:
            mean_loss = np.mean(episode_losses) if episode_losses else None
            print(f"\n--- Épisode {episode}/{EPISODES} ---")
            print(f"  Nombre de pas: {step_count}")
            print(f"  Récompense totale: {episode_reward:.2f}")
            if mean_loss is None:
                print("  Loss DQN : mémoire en cours de remplissage")
            else:
                print(f"  Loss DQN moyenne : {mean_loss:.4f}")
            print(f"  Epsilon actuel: {epsilon:.4f}")

    print("\n=============================================")
    print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=============================================")
    model_path = Path(__file__).resolve().parents[1] / "models" / "dqn_model.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    print(f"Modèle sauvegardé dans : {model_path}")
    return total_rewards


def plot_rewards(rewards):
    output_path = Path(__file__).resolve().parents[1] / "assets" / "training_history.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = range(1, len(rewards) + 1)

    plt.figure()
    plt.plot(episodes, rewards, alpha=0.35, label="Récompense par épisode")

    window = 50
    if len(rewards) >= window:
        moyenne = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            range(window, len(rewards) + 1),
            moyenne,
            label=f"Moyenne sur {window} épisodes"
        )

    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.title("Performance de l'entraînement DQN")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Graphique enregistré dans : {output_path}")
    plt.show()

if __name__ == '__main__':
    # Pour éviter les erreurs de dépendance non installées, il est recommandé d'installer : 
    # pip install torch numpy pygame
    
    random.seed(42) # Seed pour la reproductibilité
    np.random.seed(42)
    torch.manual_seed(42)

    rewards = train_agent()
    plot_rewards(rewards)