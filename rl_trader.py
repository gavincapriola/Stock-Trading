import argparse
import datetime
import itertools
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


# APPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
    """Get the data from the CSV file.
    
    Returns:
        T x 3 list of stock prices
        each row is a different stock
        0 = APPL
        1 = MSI
        2 = SBUX
    """
    df = pd.read_csv('appl_msi_sbux.csv')
    return df.values


class ReplayBuffer:
    """Replay buffer to store transitions that the agent observes.
    
    This is instead of storing transitions in the experience tuple.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size
    
    def store(self, obs, act, rew, next_obs, done):
        """Store a transition in the replay buffer."""
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample_batch(self, batch_size=32):
        """Sample a batch of transitions from the replay buffer."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def get_scaler(env):
    """Returns a scaler object to scale the states."""
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def maybe_make_dir(directory):
    """Makes a directory if it doesn't already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


class MLP(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden_layers=1, hidden_dim=32):
        super(MLP, self).__init__()
        
        M = n_inputs
        self.layers = []
        for _ in range(n_hidden_layers):
            layer = nn.Linear(M, hidden_dim)
            M = hidden_dim
            self.layers.append(layer)
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(M, n_actions))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


def predict(model, np_states):
    """Returns the action to take given a state."""
    with torch.no_grad():
        inputs = torch.from_numpy(np_states.astype(np.float32))
        output = model(inputs)
        # print(f'output: {output}')
        return output.numpy()
    

def train_one_step(model, criterion, optimizer, inputs, targets):
    # convert to tensors
    inputs = torch.from_numpy(inputs.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # backward and optimize
    loss.backward()
    optimizer.step()
