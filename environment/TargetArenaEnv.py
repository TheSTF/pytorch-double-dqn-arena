import random

import numpy as np
import pygame


class TargetArenaEnv:
    """
    Arène 2D (10x10) avec murs internes solides, obstacles mortels et cible.
    - État : vecteur de 14 floats normalisés.
    - Actions : 0=Haut, 1=Bas, 2=Gauche, 3=Droite.
    """

    def __init__(self, size: int = 10, cell_size: int = 50, render_mode: bool = False, fps: int = 10):
        self.size = size
        self.cell_size = cell_size
        self.hud_height = 44
        self.width = size * cell_size
        self.height = size * cell_size + self.hud_height
        self.render_mode = render_mode
        self.fps = fps
        self.quit_requested = False

        # Murs internes solides (bloquent le mouvement)
        self.walls = {
            (3, 1), (3, 2), (3, 3), (3, 4),
            (6, 5), (6, 6), (6, 7), (6, 8),
            (1, 7), (2, 7),
        }

        # Obstacles dangereux (fin d'épisode immédiate)
        self.obstacles = {
            (1, 2), (5, 2), (4, 7), (8, 3),
        }

        self.blocked_cells = self.walls | self.obstacles
        self.free_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.blocked_cells
        ]

        self.screen = None
        self.clock = None
        self.font = None
        if self.render_mode:
            self._init_pygame()

        self.agent_pos = (0, 0)
        self.target_pos = (self.size - 1, 0)
        self.reset()

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("DQN - Arène 2D (Murs & Obstacles)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

    def reset(self) -> np.ndarray:
        """Place l'agent et la cible sur deux cases libres distinctes."""
        self.agent_pos, self.target_pos = random.sample(self.free_cells, 2)
        return self._get_state()

    def _wall_distance(self, dx: int, dy: int) -> float:
        """Distance normalisée (0 = collé au mur) dans une direction donnée."""
        x, y = self.agent_pos
        dist = 0
        while True:
            x += dx
            y += dy
            if x < 0 or x >= self.size or y < 0 or y >= self.size or (x, y) in self.walls:
                break
            dist += 1
        return dist / (self.size - 1)

    def _get_state(self) -> np.ndarray:
        ax, ay = self.agent_pos
        tx, ty = self.target_pos
        denom = float(self.size - 1)

        # 1. Coordonnées normalisées + vecteur relatif vers la cible (6 valeurs)
        ax_n, ay_n = ax / denom, ay / denom
        tx_n, ty_n = tx / denom, ty / denom
        dx_n, dy_n = (tx - ax) / denom, (ty - ay) / denom

        # 2. Distances aux murs dans les 4 directions (4 valeurs)
        wall_sensors = [
            self._wall_distance(0, -1),  # Haut
            self._wall_distance(0, 1),   # Bas
            self._wall_distance(-1, 0),  # Gauche
            self._wall_distance(1, 0),   # Droite
        ]

        # 3. Présence d'un obstacle mortel sur les 4 cases adjacentes (4 valeurs)
        hazard_sensors = [
            1.0 if (ax, ay - 1) in self.obstacles else 0.0,
            1.0 if (ax, ay + 1) in self.obstacles else 0.0,
            1.0 if (ax - 1, ay) in self.obstacles else 0.0,
            1.0 if (ax + 1, ay) in self.obstacles else 0.0,
        ]

        return np.array(
            [ax_n, ay_n, tx_n, ty_n, dx_n, dy_n, *wall_sensors, *hazard_sensors],
            dtype=np.float32,
        )

    def step(self, action: int):
        movements = {
            0: (0, -1),  # Haut
            1: (0, 1),   # Bas
            2: (-1, 0),  # Gauche
            3: (1, 0),   # Droite
        }
        dx, dy = movements[action]
        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        old_dist = abs(tx - ax) + abs(ty - ay)
        nx, ny = ax + dx, ay + dy

        done = False
        info = {"result": "move"}

        # 1. Collision avec une bordure ou un mur interne -> bloqué + pénalité
        out_of_bounds = nx < 0 or nx >= self.size or ny < 0 or ny >= self.size
        if out_of_bounds or (nx, ny) in self.walls:
            reward = -1.0
            info["result"] = "wall"

        # 2. Collision avec un obstacle rouge -> échec immédiat
        elif (nx, ny) in self.obstacles:
            self.agent_pos = (nx, ny)
            reward = -10.0
            done = True
            info["result"] = "obstacle"

        # 3. Cible atteinte -> succès
        elif (nx, ny) == self.target_pos:
            self.agent_pos = (nx, ny)
            reward = 20.0
            done = True
            info["result"] = "target"

        # 4. Déplacement libre -> coût par pas + léger guidage potentiel
        else:
            self.agent_pos = (nx, ny)
            new_dist = abs(tx - nx) + abs(ty - ny)
            reward = -0.1 + 0.1 * (old_dist - new_dist)

        return self._get_state(), reward, done, info

    def render(self, hud_text: str = ""):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return

        self.screen.fill((24, 26, 32))

        # Barre d'info supérieure (HUD)
        pygame.draw.rect(self.screen, (16, 18, 22), (0, 0, self.width, self.hud_height))
        if hud_text:
            label = self.font.render(hud_text, True, (230, 235, 240))
            self.screen.blit(label, (12, 12))

        # Grille
        for y in range(self.size):
            for x in range(self.size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    self.hud_height + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, (42, 46, 56), rect, 1)

                if (x, y) in self.walls:
                    pygame.draw.rect(self.screen, (110, 118, 132), rect.inflate(-4, -4), border_radius=6)

                elif (x, y) in self.obstacles:
                    inner = rect.inflate(-8, -8)
                    pygame.draw.rect(self.screen, (215, 65, 65), inner, border_radius=6)
                    pygame.draw.line(self.screen, (255, 220, 220), inner.topleft, inner.bottomright, 2)
                    pygame.draw.line(self.screen, (255, 220, 220), inner.topright, inner.bottomleft, 2)

        # Cible (vert)
        tx, ty = self.target_pos
        target_center = (
            tx * self.cell_size + self.cell_size // 2,
            self.hud_height + ty * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(self.screen, (45, 210, 115), target_center, self.cell_size // 3)

        # Agent (bleu)
        ax, ay = self.agent_pos
        agent_center = (
            ax * self.cell_size + self.cell_size // 2,
            self.hud_height + ay * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(self.screen, (65, 160, 255), agent_center, self.cell_size // 3)
        pygame.draw.circle(self.screen, (220, 240, 255), agent_center, self.cell_size // 3, 2)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.render_mode and self.screen is not None:
            pygame.quit()
            self.screen = None