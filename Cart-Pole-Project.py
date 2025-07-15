# Patch numpy missing alias for Gym compatibility
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 100
MAX_STEPS = 50000

# Environment setup
env = gym.make('CartPole-v1', render_mode='human')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Q-Network definition
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_vals = self.forward(obs_t)
            return q_vals.argmax(dim=1).item()

# Initialize networks and optimizer
policy_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# Replay buffer
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque(maxlen=100)

# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return policy_net.act(state)

# Fill replay buffer with random transitions
obs, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.append((obs, action, reward, next_obs, done))
    obs = next_obs if not done else env.reset()[0]

# Training loop
obs, _ = env.reset()
episode_reward = 0

for step in range(1, MAX_STEPS + 1):
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    action = select_action(obs, epsilon)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.append((obs, action, reward, next_obs, done))
    obs = next_obs
    episode_reward += reward

    if done:
        obs, _ = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0

    # Check solved
    if len(rew_buffer) == 100 and sum(rew_buffer)/100 >= 195:
        print(f"Solved at step {step}!")
        break

    # Sample batch and train
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards_batch, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards_t = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards_t + GAMMA * (1 - dones_t) * max_next_q

    q_values = policy_net(states).gather(1, actions)
    loss = nn.functional.mse_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if step % 1000 == 0:
        avg_rew = sum(rew_buffer)/len(rew_buffer) if rew_buffer else 0
        print(f"Step: {step}, Avg Reward (last 100): {avg_rew:.2f}")

# Evaluation with rendering
eval_obs, _ = env.reset()
done = False
print("Starting evaluation...")
while not done:
    env.render()
    action = policy_net.act(eval_obs)
    eval_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.02)

env.close()
