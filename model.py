import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import settings as st


class QNetwork(nn.Module):
    """
    Deep Q-Network (DQN) neural network for reinforcement learning.

    This network consists of three fully connected layers.

    Args:
        None

    Attributes:
        linear1 (nn.Linear): First fully connected layer.
        linear2 (nn.Linear): Second fully connected layer.

    Methods:
        forward(x): Forward pass of the neural network.
        save(file_name='model.pth'): Save the model's state_dict to a file.
    """

    def __init__(self):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features=11, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=3)

    def forward(self, x):
        """
        Forward pass of the neural network.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        """
        Save the model's state_dict to a file.

            Args:
                file_name (str): Name of the file to save the model to.

            Returns:
                None
        """
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"[INFO] Saved model successfully, located: {file_path}")


class QTrainer:
    """
    QTrainer class for training a Q-learning agent.

        Attributes:
            policy_net (torch.nn.Module): The policy network for Q-learning.
            target_net (torch.nn.Module): The target network used in the Q-learning update.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model's parameters.
            criterion (torch.nn.modules.loss._Loss): The loss function used for training.

        Methods:
            __init__(self, model): Initializes a QTrainer instance.
            train(self, state, action, reward, next_state, done): Trains the Q-learning agent.
            update_target_net(self): Updates the target network to match the policy network.
    """

    def __init__(self, model):
        """
        Initializes a QTrainer instance.

            Parameters:
                model (torch.nn.Module): The Q-learning model.
        """
        self.policy_net = model
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = optim.Adam(model.parameters(), lr=st.LR)
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        """
        Trains the Q-learning agent using the provided experiences.

            Parameters:
                state (torch.Tensor): The current state (shape: (BATCH, 20)).
                action (torch.Tensor): The taken actions (shape: (BATCH, 4)).
                reward (torch.Tensor): The immediate rewards (shape: (BATCH, 1)).
                next_state (torch.Tensor): The next state (shape: (BATCH, 20)).
                done (torch.Tensor): Binary indicators for episode completion (shape: (BATCH, 1)).

            Returns:
                float: The loss value incurred during the training step.

            Notes:
            - Shapes: state = (BATCH, 20), action = (BATCH, 4), reward = (BATCH, 1), next_state = (BATCH, 20), done = (BATCH, 1).
            - The method applies the Q-learning algorithm using the Bellman equation.
            - Computes Q-values for the current and next states, and updates the model's parameters to minimize the Mean Squared Error (MSE) loss.
            - The discount factor (st.GAMMA) is used to weigh future rewards.
        """
        predicted_q_values = self.policy_net(state)  # (BATCH, 4)
        pred_action_q_values = torch.gather(
            predicted_q_values, 1, torch.argmax(action, dim=1).unsqueeze(1)
        )
        target_q_values_next_state = self.target_net(next_state)  # (BATCH, 4)
        temp = target_q_values_next_state.max(dim=1)[0].unsqueeze(
            1
        )
        # Bellman Equation: Q(s, a) = R + γ * max(Q(s', a'))
        # - Q(s, a): Q-value for state s and action a.
        # - R: Immediate reward observed in the current step.
        # - γ: Discount factor for future rewards.
        # - max(Q(s', a')): Maximum Q-value for the next state s' and all possible actions a'.

        # Compute the target Q-value based on the Bellman equation
        target_q_values = reward.unsqueeze(1) + (
            1 - done.unsqueeze(1)
        ) * st.GAMMA * target_q_values_next_state.max(dim=1)[0].unsqueeze(
            1
        )  # target_q_values shape = (BATCH, 1)

        # Zero the gradients
        self.optimizer.zero_grad()

        # Compute the MSE loss between predicted and target Q-values
        loss = self.criterion(pred_action_q_values, target_q_values)

        # Backpropagation
        loss.backward()

        # Update the model's parameters
        self.optimizer.step()

        return loss.item()  # Return the loss value for analysis

    def update_target_net(self):
        """
        Updates the target network to match the policy network.
        """
        self.target_net = copy.deepcopy(self.policy_net)
