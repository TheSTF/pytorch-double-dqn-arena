import numpy as np

class SimpleEnv:
    """
    Un environnement simulé basé sur une grille pour tester un agent DQN.
    L'état est la position (x, y) de l'agent dans un espace N x M.
    Les actions sont les déplacements : Haut, Bas, Gauche, Droite, ou Aucune.
    """

    def __init__(self, size=5):
        """
        Initialise l'environnement.
        :param size: Taille de la grille (size x size).
        """
        self.size = size
        # L'état est une représentation vectorielle [x, y]
        # Nous utiliserons le tuple (x, y) pour simplifier la logique et la conversion en numpy array dans step/reset.

    def reset(self):
        """
        Réinitialise l'environnement à un état initial aléatoire.
        Retourne l'état initial (observation).
        """
        # Position initiale : coin supérieur gauche par défaut ou aléatoire
        self._current_state = np.array(
            [np.random.randint(0, self.size), np.random.randint(0, self.size)],
            dtype=np.float32
        )
        return self._current_state

    def step(self, action: int):
        """
        Exécute une action depuis l'état actuel.
        :param action: Un entier représentant l'action (0=Haut, 1=Bas, 2=Gauche, 3=Droite, 4=Inactif).
        :return: tuple contenant (nouvel_etat, recompense, done)
        """
        state = self._current_state
        new_x, new_y = state

        # Déterminer le déplacement en fonction de l'action
        movements = {
            0: (0, -1),  # Haut (diminuer y)
            1: (0, 1),   # Bas (augmenter y)
            2: (-1, 0),  # Gauche (diminuer x)
            3: (1, 0),   # Droite (augmenter x)
            4: (0, 0)    # Inactif
        }

        dx, dy = movements[action]

        # Calcul de la nouvelle position brute
        next_x_raw = new_x + dx
        next_y_raw = new_y + dy

        # Gestion des limites du monde (clamping)
        next_x = np.clip(next_x_raw, 0, self.size - 1).astype(np.float32)
        next_y = np.clip(next_y_raw, 0, self.size - 1).astype(np.float32)

        new_state = np.array([next_x, next_y]).astype(np.float32)

        # Calcul de la récompense et du terminal (ici, atteindre le coin opposé au départ ou rester bloqué)
        reward = -1.0  # Pénalité par pas pour encourager l'efficacité
        done = False
        
        # Exemple simple de condition d'arrêt : si on atteint un coin prédéfini (ex: bas-droite)
        if next_x >= self.size - 1 and next_y <= 0: # Simplification de la cible pour l'exemple
             reward = 10.0
             done = True

        self._current_state = new_state
        return new_state, reward, done


    def _update_internal_state(self, state):
        """Met à jour l'état interne après un reset ou une étape réussie."""
        self._current_state = state

    @property
    def current_state(self):
        return self._current_state

# --- Exemple de test (à commenter lors de l'intégration) ---
if __name__ == '__main__':
    env = SimpleEnv(size=5)
    print("--- Test d'initialisation et de reset ---")
    initial_state = env.reset()
    print(f"État initial: {initial_state}")

    # Simuler 3 étapes : aller vers le coin bas-droit (ex: actions 3, 1, 0)
    actions_to_test = [3, 1, 0] # Droite -> Bas -> Haut
    current_state = initial_state
    for i, action in enumerate(actions_to_test):
        next_state, reward, done = env.step(action)
        print(f"\n--- Étape {i+1} (Action: {action}) ---")
        print(f"État avant: {current_state}")
        print(f"Nouvel état: {next_state}")
        print(f"Récompense: {reward}, Terminé: {done}")
        current_state = next_state

    # Réinitialiser pour un autre test
    env = SimpleEnv(size=5)
    final_state = env.reset()
    print(f"\nÉtat après reset final: {final_state}")