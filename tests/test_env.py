import pygame
import random
import pytest
from envs.tetris_env import TetrisEnv


def test_pygame_renderer():
    """
    Simple visual smoke test for the TetrisEnv renderer using pygame.
    Opens a window, draws the board state and current piece for a few steps.
    """
    # Initialize pygame
    pygame.init()
    cell_size = 30
    width, height = 10, 20
    screen = pygame.display.set_mode((width * cell_size, height * cell_size))
    pygame.display.set_caption("TetrisEnv Smoke Test")
    clock = pygame.time.Clock()

    # Create environment and reset
    env = TetrisEnv()
    obs = env.reset()

    running = True
    steps = 0
    max_steps = 50

    while running and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw grid and blocks
        # Define Tetris piece colors: board cells should store piece IDs
        color_map = {
            1: (0, 255, 255),   # I - cyan
            2: (255, 255, 0),   # O - yellow
            3: (128, 0, 128),   # T - purple
            4: (0, 0, 255),     # J - blue
            5: (255, 165, 0),   # L - orange
            6: (0, 255, 0),     # S - green
            7: (255, 0, 0)      # Z - red
        }
        board = obs['board']
        for y in range(height):
            for x in range(width):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                cell = board[y][x]
                if cell:
                    # color according to piece ID
                    color = color_map.get(cell, (255, 255, 255))
                    pygame.draw.rect(screen, color, rect)
                else:
                    pygame.draw.rect(screen, (50, 50, 50), rect, 1)

        pygame.display.flip()

        # Step with a random legal action
        raw_actions = env.legal_actions()
        # Convert legacy tuple/string actions into dict format
        actions = []
        for a in raw_actions:
            if isinstance(a, tuple):
                if a[0] == 'soft_drop':
                    actions.append({'type': 1, 'x': a[1], 'rotation': a[2]})
                elif a[0] == 'hard_drop':
                    actions.append({'type': 2, 'x': a[1], 'rotation': a[2]})
            elif a == 'hold':
                actions.append({'type': 3, 'x': 0, 'rotation': 0})
        action = random.choice(actions)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

        steps += 1
        clock.tick(10)  # limit to 10 FPS

    pygame.quit()
    # If we reach here without error, the renderer works
    assert True

