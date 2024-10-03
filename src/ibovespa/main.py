import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from gym_trading.agent import Agent
from gym_trading.enviroment import PortfolioEnv 
import pandas as pd
from gym_trading.network import ActorCritic
import argparse

def main(period):
    stock_data = f"./data/br_stocks_{period}.csv"
    index_data = f"./data/ibov_{period}.csv"

    stock_data = stock_data.set_index("Ticker")
    index_data = index_data.set_index("Ticker")
    
    min_length = min(len(stock_data), len(index_data))
    stock_data = stock_data[-min_length:]
    index_data = index_data[-min_length:]

    ### enviroment ###
    env = PortfolioEnv(stock_data, index_data)
    
    ### network ###
    HIDDEN_SIZE = 256
    actor_critic = ActorCritic(860, len(env.action_space), hidden_size=HIDDEN_SIZE)
    optimizer, scheduler =  actor_critic.set_params()

    ### agent ###
    agent = Agent(env, actor_critic, optimizer, scheduler)
    agent.train(200)
    agent.save_model(f"./models/ibovespa_{period}_2024.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, required=True, help="pre-pandemic, pandemic, post-pandemic, or all")
    args = parser.parse_args()

    main(args.period)
