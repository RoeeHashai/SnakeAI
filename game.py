import pygame
import sys
import random
import numpy as np
import settings as st
from collections import namedtuple

# Named tuple to represent a point with x and y coordinates
Point = namedtuple("Point", ["x", "y"])


class Snake:
    """Class representing the snake in the game."""

    def __init__(self):
        """Initialize snake properties."""
        self.head = Point(st.WIDTH // 2, st.HEIGHT // 2 + st.BLOCK_SIZE)
        self.body = [
            self.head,
            Point(self.head.x, self.head.y + st.BLOCK_SIZE),
            Point(self.head.x, self.head.y + (2 * st.BLOCK_SIZE)),
        ]
        self.direction = st.Direction.UP
        self.lenght = 3
        self.color = st.GREEN

    def move(self):
        """Update the snake's position."""
        # Determine what is the current direction of the snake
        if self.direction == st.Direction.RIGHT:
            next_head = Point(self.head.x + st.BLOCK_SIZE, self.head.y)
        elif self.direction == st.Direction.LEFT:
            next_head = Point(self.head.x - st.BLOCK_SIZE, self.head.y)
        elif self.direction == st.Direction.UP:
            next_head = Point(self.head.x, self.head.y - st.BLOCK_SIZE)
        elif self.direction == st.Direction.DOWN:
            next_head = Point(self.head.x, self.head.y + st.BLOCK_SIZE)

        # Update the head of the snake and insert to the bode list
        self.head = next_head
        self.body.insert(0, next_head)

        # If the snake didn't eat the food, the last block is poped out of body list
        if len(self.body) > self.lenght:
            self.body.pop()

    def is_self_collision(self):
        """Check if the snake collides with itself."""
        return self.head in self.body[1:]

    def set_direction(self, direction):
        """Set the direction of the snake."""
        self.direction = direction

    def grow(self):
        """Increase the length of the snake."""
        self.lenght += 1

    def reset(self):
        """Reset the snake to its initial state."""
        self.head = Point(st.WIDTH // 2, st.HEIGHT // 2 + st.BLOCK_SIZE)
        self.body = [
            self.head,
            Point(self.head.x, self.head.y + st.BLOCK_SIZE),
            Point(self.head.x, self.head.y + (2 * st.BLOCK_SIZE)),
        ]
        self.direction = st.Direction.UP
        self.lenght = 3

    def render(self, surface):
        """
        Render the snake on the given surface.
                Parameters:
                    surface (Surface): A gui surface
        """
        for block in self.body:
            pygame.draw.rect(
                surface, self.color, (block.x, block.y, st.BLOCK_SIZE, st.BLOCK_SIZE)
            )

    def get_head(self):
        """Get the head of the snake."""
        return self.head
    
    def get_direction(self):
        """Get the direction of the snake."""
        return self.direction
    
    def get_body(self):
        """Get the body of the snake."""
        return self.body


class Food:
    def __init__(self, snake):
        """
        Initialize food properties.
            Parameters:
                    snake (Snake): The snake of the game
        """
        self.snake = snake
        self.position = self.gen_random_position()
        self.color = st.RED

    def gen_random_position(self):
        """Randomize the food position."""
        x = random.randint(0, ((st.WIDTH - st.BLOCK_SIZE) // st.BLOCK_SIZE)) * st.BLOCK_SIZE
        y = (random.randint(0, ((st.HEIGHT - st.BLOCK_SIZE) // st.BLOCK_SIZE)) * st.BLOCK_SIZE) + st.BLOCK_SIZE   
        pos = Point(x, y)
        if pos in self.snake.body:
            self.gen_random_position()
        return pos

    def randomize_position(self):
        """Randomize the position of the food."""
        self.position = self.gen_random_position()

    def is_eaten(self):
        """Check if the food is eaten by the snake."""
        return self.position == self.snake.get_head()

    def render(self, surface):
        """
        Render the food on the given surface.
                Parameters:
                    surface (Surface): A gui surface
        """
        pygame.draw.rect(
            surface,
            self.color,
            (self.position.x, self.position.y, st.BLOCK_SIZE, st.BLOCK_SIZE),
        )
        
    def get_position(self):
        """Get the position of the food."""
        return self.position


class Game:
    def __init__(self):
        """Initialize the game properties."""
        self.snake = Snake()
        self.food = Food(self.snake)
        self.score = 0
        pygame.init()
        self.surface = pygame.display.set_mode((st.WIDTH, st.HEIGHT + st.BLOCK_SIZE))
        self.score_font = pygame.font.Font(None, 33)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("SnakeAI")
        self.count_step = 0

    def reset_game(self):
        """Reset the game to its initial state."""
        self.score = 0
        self.food.randomize_position()
        self.snake.reset()
        self.count_step =0


    def set_snake_direction(self, action):
        """
        Set the direction of the snake based on the given action.

        Parameters:
            action (numpy.ndarray): An array representing the chosen action.

        Note:
            The action array is expected to be one of the following:
            - [1, 0, 0]: Set direction to RIGHT.
            - [0, 1, 0]: Set direction to CONTINUE.
            - [0, 0, 1]: Set direction to LEFT.
        """
        # Check the action array and set the snake direction accordingly
        direction_values = list(st.Direction) # [RIGHT, DOWN, LEFT, UP]
        current_direction = self.snake.get_direction()
        if np.array_equal([0,1,0], action):
            new_direction = current_direction
        elif np.array_equal([0,0,1], action): # right turn
            new_direction = direction_values[(direction_values.index(current_direction) + 1) % len(direction_values)]
        elif np.array_equal([1,0,0], action): # left turn
            new_direction = direction_values[(direction_values.index(current_direction) - 1) % len(direction_values)]
            
        self.snake.set_direction(new_direction)        

    def render(self):
        """Render the game on the screen."""
        # 1. set the background
        self.surface.fill(st.BLACK)

        # 2. create the score bar
        pygame.draw.rect(self.surface, st.GRAY, (0, 0, st.WIDTH, st.BLOCK_SIZE))
        score_text = self.score_font.render(f"Score: {self.score}", True, st.WHITE)
        self.surface.blit(score_text, (st.WIDTH // 2 - (2*st.BLOCK_SIZE), 0))

        # 3. render the food on the surface
        self.food.render(self.surface)

        # 4. render the snake on the surface
        self.snake.render(self.surface)

        # 5. update the gui contents of the entire display window
        pygame.display.flip()

    def is_collision(self, pt = None):
        """
        Check if there is a collision in the game.

            Parameters:
                pt (Point): The point to check for collision. If None, check the snake's head.

            Returns:
                bool: True if there is a collision, False otherwise.
        """ 
        if pt is None:
            pt = self.get_snake().get_head()
        return (
            (pt in self.snake.get_body()[1:])
            or (pt.x >= st.WIDTH or pt.x < 0)
            or (
                pt.y >= st.HEIGHT + st.BLOCK_SIZE
                or pt.y < st.BLOCK_SIZE
            )
        )

    def play_step(self, action):
        """
        Play a step in the game.

            Parameters:
                action (numpy.ndarray): An array representing the chosen action.

            Returns:
                game_over (bool): True if the game is over, False otherwise.
                reward (int): The reward obtained in this step.
                score (int): The current score.
        """
        self.count_step +=1
        reward = 0
        game_over = False
        for event in pygame.event.get():
            # user wants to exit
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # render the screen show the current state
        self.render()
        self.clock.tick(st.FPS)
        
        # receive the action and update the direction of the snake
        self.set_snake_direction(action)

        # apply the move on the snake
        self.snake.move()
        
        # check if the food was eaten
        if self.food.is_eaten():
            self.score += 1
            self.snake.grow()
            self.food.randomize_position()
            reward = 10

        # check if an collision occurred
        if self.is_collision() or self.count_step > len(self.snake.get_body()) * 100:
            reward = -10
            game_over = True
            
        return game_over, reward, self.score

    def get_snake(self):
        """Get the snake object."""
        return self.snake
    
    def get_food(self):
        """Get the food object."""
        return self.food
    

