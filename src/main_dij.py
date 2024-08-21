from gym_trading.agent import Agent
from gym_trading.enviroment import PortfolioEnv 
import pandas as pd
from gym_trading.network import ActorCritic

def main():
    stock_data = pd.read_csv("./data/dji_stocks_adj_close.csv")
    stock_data = stock_data.set_index("Ticker")
    
    index_data = pd.read_csv("./data/dow_jones_index_close.csv")
    index_data = index_data.set_index("Ticker")
    index_data.drop(columns=["Unnamed: 0"], inplace=True)
    
    min_length = min(len(stock_data), len(index_data))
    stock_data = stock_data[-min_length:]
    index_data = index_data[-min_length:]

    ### enviroment ###
    env = PortfolioEnv(stock_data, index_data)
    
    ### network ###
    HIDDEN_SIZE = 256
    # print(env.observation_space.shape, len(env.action_space))
    actor_critic = ActorCritic(300, len(env.action_space), hidden_size=HIDDEN_SIZE)
    optimizer, scheduler =  actor_critic.set_params()

    ### agent ###
    agent = Agent(env, actor_critic, optimizer, scheduler)
    agent.train(200)
    agent.save_model("./models/actor_critic_dow_jones.pth")

if __name__ == "__main__":
    main()
