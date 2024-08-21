import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from gym_trading.agent import Agent
from gym_trading.enviroment import PortfolioEnv 
import pandas as pd
from gym_trading.network import ActorCritic

def main():
    stock_data = pd.read_csv("./data/ibovespa_pandemic_data.csv")
    # stock_data.drop(columns=["Unnamed: 0"], inplace=True)
    # stock_data = stock_data.sample(frac=0.05, random_state=42)
    stock_data = stock_data.set_index("Ticker")
    index_data = stock_data.copy()
    
    index_data = pd.DataFrame(index_data["^BVSP.4"])
    stock_data = stock_data.drop(columns=["^BVSP.4"])

    min_length = min(len(stock_data), len(index_data))
    stock_data = stock_data[-min_length:]
    index_data = index_data[-min_length:]

    ### enviroment ###
    env = PortfolioEnv(stock_data, index_data)
    
    ### network ###
    HIDDEN_SIZE = 256
    # print(env.observation_space.shape, len(env.action_space))
    actor_critic = ActorCritic(2850, len(env.action_space), hidden_size=HIDDEN_SIZE)
    optimizer, scheduler =  actor_critic.set_params()

    ### agent ###
    agent = Agent(env, actor_critic, optimizer, scheduler)
    agent.train(200)
    agent.save_model("./models/ibovespa_pandemic.pth")

if __name__ == "__main__":
    main()
