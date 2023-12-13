from enum import Enum

# game constants
WIDTH, HEIGHT = 800, 600
FPS = 100


class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# single snake block size
BLOCK_SIZE = 20

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

# Hyperparametes
LR = 0.001
GAMMA = 0.9

# Agent constants
MAX_MEMORY = 100000
BATCH_SIZE = 1000
EPSILON = 1
TARGET_UPDATE_FREQUENCY = 100
