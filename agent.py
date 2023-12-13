import torch
import numpy as np
import random
from collections import deque
from model import QNetwork, QTrainer
from game import Game, Point
import settings as st


class Agent:
    def __init__(self, enviroment):
        """
        Initialize the Agent.

            Parameters:
            - enviroment: The environment in which the agent operates.
        """
        self.enviroment = enviroment
        self.model = QNetwork()
        self.trainer = QTrainer(self.model)
        self.replay_buffer = deque(maxlen=st.MAX_MEMORY)
        self.episode_buffer = deque(maxlen=st.MAX_MEMORY)
        self.count_episodes = 0
        self.count_train_step = 0
        self.epsilon = st.EPSILON
        self.target_update_frequency = st.TARGET_UPDATE_FREQUENCY
        self.highest_score = 0

    def get_state(self):
        """
        Get the current state of the environment.

            Returns:
            - np.array: An array representing the state.
        """
        # prepare the data for the NN
        head = self.enviroment.get_snake().get_head()
        food = self.enviroment.get_food().get_position()
        next_move_r = Point(head.x + st.BLOCK_SIZE, head.y)
        next_move_l = Point(head.x - st.BLOCK_SIZE, head.y)
        next_move_u = Point(head.x, head.y - st.BLOCK_SIZE)
        next_move_d = Point(head.x, head.y + st.BLOCK_SIZE)

        # Determine the direction of the snake
        is_dir_r = self.enviroment.get_snake().get_direction() == st.Direction.RIGHT
        is_dir_l = self.enviroment.get_snake().get_direction() == st.Direction.LEFT
        is_dir_u = self.enviroment.get_snake().get_direction() == st.Direction.UP
        is_dir_d = self.enviroment.get_snake().get_direction() == st.Direction.DOWN

        state = [
            # Is there a danger ahead ==================================
            (is_dir_r and self.enviroment.is_collision(next_move_r))
            or (is_dir_l and self.enviroment.is_collision(next_move_l))
            or (is_dir_u and self.enviroment.is_collision(next_move_u))
            or (is_dir_d and self.enviroment.is_collision(next_move_d)),
            # Is there a danger on the right ============================
            (is_dir_r and self.enviroment.is_collision(next_move_d))
            or (is_dir_l and self.enviroment.is_collision(next_move_u))
            or (is_dir_u and self.enviroment.is_collision(next_move_r))
            or (is_dir_d and self.enviroment.is_collision(next_move_l)),
            # Is there a danger on the left ============================
            (is_dir_r and self.enviroment.is_collision(next_move_u))
            or (is_dir_l and self.enviroment.is_collision(next_move_d))
            or (is_dir_u and self.enviroment.is_collision(next_move_l))
            or (is_dir_d and self.enviroment.is_collision(next_move_r)),
            # Direction of the snake (hot codded) ======================
            is_dir_r,
            is_dir_l,
            is_dir_u,
            is_dir_d,
            # Food location in a realation to the head (hot codded) ====
            food.x < head.x,  # food left
            food.x > head.x,  # food right
            food.y < head.y,  # food up
            food.y > head.y,  # food down
        ]  # Total of 11 input features ==================================

        return np.array(state, dtype=int)

    def _convert_pred_action(self, action):
        """
        Convert predicted action to the format expected by the game.

            Parameters:
            - action: The predicted action.

            Returns:
            - torch.Tensor: The converted action.
        """
        converted_action = torch.zeros_like(action)
        converted_action[torch.argmax(action, dim=0).item()] = 1
        return converted_action

    def choose_action(self, state):
        """
        Choose an action for the current state. (exploration and exploitation)


        Parameters:
        - state: The current state.

        Returns:
        - torch.Tensor: The chosen action.
        """
        self.epsilon = max(0.01, self.epsilon - 0.0001)
        if torch.rand(1).item() < self.epsilon:
            rand_action = random.randint(0, 2)
            action = torch.zeros(3)
            action[rand_action] = 1
        else:
            pred_action = self.model(torch.tensor(state, dtype=torch.float32))
            action = torch.zeros_like(pred_action)
            action[torch.argmax(pred_action, dim=0).item()] = 1
        return action

    def save_to_replay_buffer(self, state, action, reward, next_state, done):
        """
        Save a transition to the replay buffer.

            Parameters:
            - state: The current state.
            - action: The chosen action.
            - reward: The received reward.
            - next_state: The next state.
            - done: True if the episode is done, False otherwise.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_replay_buffer(self):
        """
        Train the agent using experiences from the replay buffer.

            Returns:
            - float: The loss during training.
        """
        if len(self.replay_buffer) < st.BATCH_SIZE:
            sample = self.replay_buffer
        else:
            sample = random.sample(self.replay_buffer, st.BATCH_SIZE)

        state_tup, action_tup, reward_tup, next_state_tup, done_tup = zip(*sample)
        state_tensor = torch.tensor(state_tup, dtype=torch.float32)
        action_tensor = torch.stack(action_tup)
        reward_tensor = torch.tensor(reward_tup, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state_tup, dtype=torch.float32)
        done_tensor = torch.tensor(done_tup, dtype=torch.float32)

        loss = self.trainer.train(
            state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
        )
        return loss

    def train_single_step(
        self, cur_env_state, action, reward, next_env_state, game_over
    ):
        """
        Train the agent using a single step of experience.

            Parameters:
            - cur_env_state: The current state.
            - action: The chosen action.
            - reward: The received reward.
            - next_env_state: The next state.
            - game_over: True if the episode is done, False otherwise.

            Returns:
            - float: The loss during training.
        """
        state_tensor = torch.tensor(cur_env_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = action.unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_env_state, dtype=torch.float32).unsqueeze(
            0
        )
        done_tensor = torch.tensor(game_over, dtype=torch.float32).unsqueeze(0)

        loss = self.trainer.train(
            state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
        )
        return loss

    def play(self):
        """
        Start playing the game using the trained model.
        """
        while True:
            cur_env_state = self.get_state()
            action = self.choose_action(cur_env_state)
            game_over, reward, score = self.enviroment.play_step(action)
            next_env_state = self.get_state()
            self.save_to_replay_buffer(
                cur_env_state, action, reward, next_env_state, game_over
            )
            self.count_train_step += 1
            if self.count_train_step % self.target_update_frequency == 0:
                self.trainer.update_target_net()
            self.train_single_step(
                cur_env_state, action, reward, next_env_state, game_over
            )
            if game_over:
                self.enviroment.reset_game()
                self.count_episodes += 1

                loss = self.train_replay_buffer()
                print(
                    f"[INFO] Episode {self.count_episodes}, Highest Score {self.highest_score}, Current Score: {score}, Loss: {loss}"
                )
                if self.highest_score < score:
                    print(f"[INFO] New high score. saving model...")
                    self.model.save()
                    self.highest_score = score
