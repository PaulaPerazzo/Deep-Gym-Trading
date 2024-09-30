import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from gym_trading.agent import Agent
from gym_trading.enviroment import PortfolioEnv 
import pandas as pd
from gym_trading.network import ActorCritic
import argparse

def main(period):
    if period == "pre-pandemic":
        stock_data = pd.read_csv("./data/sp_stocks_pre_pandemic.csv")
        index_data = pd.read_csv("./data/sp500_index_pre_pandemic.csv")
    
    elif period == "pandemic":
        stock_data = pd.read_csv("./data/sp_stocks_pandemic.csv")
        index_data = pd.read_csv("./data/sp500_index_pandemic.csv")
    
    elif period == "post-pandemic":
        stock_data = pd.read_csv("./data/sp_stocks_post_pandemic.csv")
        index_data = pd.read_csv("./data/sp500_index_post_pandemic.csv")
    
    elif period == "all":
        stock_data = pd.read_csv("./data/sp_stocks.csv")
        index_data = pd.read_csv("./data/sp500.csv")

    stock_data = stock_data.set_index("Ticker")
    index_data = index_data.set_index("Ticker")
    
    min_length = min(len(stock_data), len(index_data))
    stock_data = stock_data[-min_length:]
    index_data = index_data[-min_length:]

    ### enviroment ###
    env = PortfolioEnv(stock_data, index_data)
    
    ### network ###
    HIDDEN_SIZE = 256
    actor_critic = ActorCritic(4980, len(env.action_space), hidden_size=HIDDEN_SIZE)
    optimizer, scheduler =  actor_critic.set_params()

    ### agent ###
    agent = Agent(env, actor_critic, optimizer, scheduler)
    agent.train(200)
    agent.save_model(f"./models/sp_500_{period}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, required=True, help="pre-pandemic, pandemic, post-pandemic, or all")
    args = parser.parse_args()

    main(args.period)
