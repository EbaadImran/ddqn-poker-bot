import random

from gym import Env

from gym_env.enums import Action
from gym_env.env import HoldemTable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F

from collections import deque, namedtuple

ALL_ACTIONS = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT, Action.RAISE_2POT}
GAMMA = 1.0
TAU = 0.005

class DQN(nn.Module):
    # Architecture of our DQN
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    # Forward vector x through our NN
    def forward(self, x):
        return self.network(x)

# Transition objects for our replay memory
class Transition:
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def total_reward(self):
        return sum(transition.reward for transition in self.memory)

    def rewards_div_hands(self):
        if len(self.memory) == 0:
            return 0  # Prevent division by zero
        return self.total_reward() / len(self.memory)

    def __len__(self):
        return len(self.memory)

# The actual agent playing poker
class PokerAgent:
    def __init__(self, environment: HoldemTable, name="Ebaad", load_model=None, model_path=None):
        num_env = environment.observation_space[0]
        num_actions = environment.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = DQN(num_env, num_actions)
        if model_path:
            self.policy_network = torch.load(model_path)
            print("Loading Model From: ", model_path)

        self.policy_network.to(device=self.device)
        self.policy_network_optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.001, amsgrad=True)

        self.target_network = DQN(num_env, num_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.to(device=self.device)

        self.env = environment
        self.memory = ReplayMemory(1000)
        self.writer = SummaryWriter()
        self.name = name

        self.batch_size = 128

    @staticmethod
    def get_action(q_values, moves):
        # Probabilistic distribution of actions based on q-values
        p = torch.softmax(q_values, dim=1)
        moves = {move.value for move in moves}
        mask = torch.ones_like(p, dtype=torch.bool)
        mask[:, list(moves)] = False

        # Zero out probabilities for illegal actions
        p[mask] = 0

        # Sample an action from the adjusted probabilities
        action = torch.multinomial(p, 1).item()

        return action

    def process_state(self, state):
        if isinstance(state, torch.Tensor):
            return torch.nan_to_num(state, nan=-1).squeeze(dim=1)
        else:
            return torch.tensor(np.array([np.nan_to_num(state, nan=-1)]), dtype=torch.float32).squeeze(dim=1).to(self.device)

    def optimize(self):
        transitions = self.memory.sample(self.batch_size)

        # Unpack the transitions to their components
        state_batch = torch.cat([transition.state for transition in transitions]).to(dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([transition.action for transition in transitions], dtype=torch.int64).unsqueeze(0).to(self.device)
        reward_batch = torch.tensor([transition.reward for transition in transitions], dtype=torch.float32).to(self.device)

        next_states = [transition.next_state for transition in transitions]
        temp_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(self.device)
        temp_next_states = torch.cat(next_states)

        # Compute Q-values for current states and actions using the policy network
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size).to(device=self.device)

        with torch.no_grad():
            next_state_values[temp_mask] = self.target_network(temp_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))

        # Optimize
        self.policy_network_optimizer.zero_grad()
        loss.backward()

        # Grad Clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.policy_network_optimizer.step()

        return loss

    def update_policy(self):
        # Retrieve parameters from both networks
        policy_state_dict = self.policy_network.state_dict()
        target_state_dict = self.target_network.state_dict()

         # Softly update the target network's parameters using those from the policy network
        for key in policy_state_dict:
            target_state_dict[key] = TAU * policy_state_dict[key] + (1 - TAU) * target_state_dict[key]

        self.target_network.load_state_dict(target_state_dict)

    def play(self, num_episodes=int(1e6), render=False, train=True):
        num_hands = 0

        sum_rewards = 0
        
        # Training Loop
        for episode in range(num_episodes):
            print(f"Episode: {episode + 1}")

            state = self.env.reset()
            state = self.process_state(state)
            player_position = None

            # A single round of simulation
            done = False
            while not done:
                print(f"Current Hand: {num_hands + 1}")

                state = self.process_state(state)
                q_values = self.policy_network(state)
                action = self.get_action(q_values, moves=self.env.legal_moves)

                # Perform an action in the environment
                step_env = self.env.step(action) # next_state, reward, done, info
                
                next_state = self.process_state(step_env[0])
                reward = step_env[1]
                done = step_env[2]
                player_data = step_env[3]['player_data']

                # Store the transition in replay memory
                self.memory.push(state, action, next_state, reward)
                position = player_data['position']

                # Check if the player's position has changed
                if player_position != position:
                    player_position = position
                    num_hands += 1

                # Training block
                if train and num_hands >= self.batch_size and num_hands % 2 == 0:
                    loss = self.optimize()
                    print(f"Loss: {loss}")

                    # Update the target network
                    self.update_policy()
                    self.writer.add_scalar("loss", scalar_value=loss, global_step=num_hands)
                    if reward != 0:
                        self.writer.add_scalar("rewards", scalar_value=reward, global_step=num_hands)
                        sum_rewards += reward
                        self.writer.add_scalar("cumulative_rewards", scalar_value=sum_rewards, global_step=num_hands)

                self.writer.add_scalar("reward/hands", scalar_value=self.memory.rewards_div_hands(), global_step=num_hands)

                state = next_state

                # Save model every 250 hands
                if num_hands % 250 == 0:
                    torch.save(self.policy_network, f"runs/model_{num_hands}_hands.pth")
                