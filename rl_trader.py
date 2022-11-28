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


class MultiStockEnv:
    """
    A 3-stock trading environment.
    State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
    """
    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3**self.n_stock)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(
            map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # calculate size of state
        self.state_dim = self.n_stock * 2 + 1

        self.reset()
        
    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()
    
    def step(self, action):
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in portfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs
    
    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand
    
    def _trade(self, action):
        # index each action
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2, 1, 0] means:
        # - buy first stock
        # - hold second stock
        # - sell third stock
        action_vec = self.action_list[action]
        
        # determine which stocks to buy or sell
        sell_index = [] # stores index of stocks to sell
        buy_index = [] # stores index of stocks to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)
                
        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will loop through each stock we want to buy,
            # and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False
