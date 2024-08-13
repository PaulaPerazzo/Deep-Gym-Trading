import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_data, index_data, window_size=10, max_shares=30):
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data
        self.index_data = index_data
        self.window_size = window_size
        self.current_step = 0
        self.action_history = []
        self.max_shares = max_shares
        self.done = False
        self.gamma = 0.99
        
        # Define action and observation space
        self.action_space = gym.spaces.MultiDiscrete([max_shares + 1] * stock_data.shape[1])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, stock_data.shape[1]), dtype=np.float32)


    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.state = self.stock_data.iloc[:self.window_size].values

        return self.state


    def step(self, action):
        # Adjust action if the sum exceeds max_shares
        if np.sum(action) > self.max_shares:
            action = self.adjust_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.stock_data):
            self.done = True

            return self.state, 0, self.done, {}

        self.state = self.stock_data.iloc[self.current_step-self.window_size:self.current_step].values
        reward = self.compute_reward(action)
        # print('action', action)
        name_action = dict(zip(self.stock_data.columns, action))
        # print('name_action', name_action)
        self.action_history.append(name_action)

        return self.state, reward, self.done, {}


    def adjust_action(self, action):
        while np.sum(action) > self.max_shares:
            # Encontra índices onde a ação é maior que zero
            positive_indices = np.where(action > 0)[0]

            if not positive_indices.size:
                break  # Se não houver ações para ajustar, saia do loop
            
            # Escolha aleatoriamente um índice para reduzir
            reduce_index = np.random.choice(positive_indices)

            action[reduce_index] -= 1

        return action


    def compute_reward(self, action):
        current_prices = self.stock_data.iloc[self.current_step].values
        previous_prices = self.stock_data.iloc[self.current_step - 1].values

        index_price_curr = self.index_data.iloc[self.current_step].values
        index_price_prev = self.index_data.iloc[self.current_step - 1].values

        action = action.reshape(1, -1)
        portfolio_return = np.dot(action, (current_prices - previous_prices) / previous_prices)
        index_return = (index_price_curr - index_price_prev) / index_price_prev

        # portfolio_volat = np.std(portfolio_return)
        # index_volat = np.std(index_return)

        # normalize rewards
        normalized_portfolio_return = portfolio_return / np.abs(portfolio_return).max()

        # Penalize for volatility
        volatility_penalty = np.std(normalized_portfolio_return) if len(self.action_history) > 1 else 0

        # Calculate current drawdown
        current_value = np.dot(action, current_prices)
        if len(self.action_history) > 1:
            max_value = max([np.dot(a, self.stock_data.iloc[step].values) for step, a in enumerate(self.action_history)])
        else:
            max_value = current_value
        drawdown_penalty = max(0, (max_value - current_value) / max_value)

        # Risk-adjusted reward with drawdown penalty
        reward = (normalized_portfolio_return - index_return) / (1 + volatility_penalty + drawdown_penalty)

        # reward = ((portfolio_return - index_return) / portfolio_volat) - 0.8 * abs(portfolio_volat - index_volat)
        # reward = portfolio_return - index_return
        # reward = normalized_portfolio_return - index_return

        # return reward
        return reward


    def render(self, mode='human', close=False):
        pass


    def print_action_history(self):
        print("Choosen actions: ")

        for step, actions in enumerate(self.action_history):
            print(f"Step {step}:")

            for stock_name, num_shares in actions.items():
                if num_shares > 0:
                    print(f"  {stock_name}: {num_shares} stocks")
