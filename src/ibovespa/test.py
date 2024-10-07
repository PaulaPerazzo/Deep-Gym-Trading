import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from ibovespa.mailer import send_email
import numpy as np
import torch
import pandas as pd
from gym_trading.agent import Agent
from gym_trading.enviroment import PortfolioEnv
from gym_trading.network import ActorCritic
from metrics.criteria import ProfitCriteria, ReturnMetrics, RiskCriteria, RiskReturnCriteria
import argparse
import csv

def main(period, i):
    stock_data = pd.read_csv("./data/br_stocks_2019.csv")
    index_data = pd.read_csv("./data/ibov_2019.csv")
    
    # Configuração da rede
    actor_critic = ActorCritic(inputs_features=860, n_actions=86, hidden_size=256)
    print("ActorCritic initialized.")
    
    if period == "pre-pandemic":
        # load data
        stock_data = pd.read_csv("./data/br_stocks_2019.csv")
        index_data = pd.read_csv("./data/ibov_2019.csv")
        
        # load trained model
        model_path = "./models/ibovespa_pre-pandemic_2024.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")

    elif period == "pandemic":
        stock_data = pd.read_csv("./data/br_stocks_2021.csv")
        index_data = pd.read_csv("./data/ibov_2021.csv")

        model_path = "./models/ibovespa_pandemic_2024.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")

    elif period == "post-pandemic":
        stock_data = pd.read_csv("./data/br_stocks_2023.csv")
        index_data = pd.read_csv("./data/ibov_2023.csv")

        model_path = "./models/ibovespa_post-pandemic_2024.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")
        
    elif period == "all":
        stock_data = pd.read_csv("./data/br_stocks_2023.csv")
        index_data = pd.read_csv("./data/ibov_2023.csv")
        
        model_path = "./models/ibovespa_all_2024.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")
    
    stock_data.set_index("Ticker", inplace=True)
    index_data.set_index("Ticker", inplace=True)

    min_length = min(len(stock_data), len(index_data))
    stock_data = stock_data[-min_length:]
    index_data = index_data[-min_length:]

    env = PortfolioEnv(stock_data, index_data)
    print("Environment initialized.")

    agent = Agent(env, actor_critic, optimizer=None, scheduler=None)
    print("Agent initialized.")

    state = env.reset().flatten() 
    done = False
    actions_taken = []
    rewards = []
    selected_stocks = []
    portfolio_values = []

    initial_portfolio_value = 0  
    current_portfolio_value = initial_portfolio_value

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = agent.decide_action(state_tensor, max_share=30).numpy().flatten()
        state, reward, done, _ = env.step(action)  
        state = state.flatten()  
        actions_taken.append(action)
        rewards.append(reward)

        stock_names = stock_data.columns
        selected = [stock_names[i] for i, num in enumerate(action) if num > 0]
        selected_stocks.append(selected)

        if env.current_step < len(env.stock_data):
            # Calculate the current portfolio value
            current_prices = env.stock_data.iloc[env.current_step].values
            current_investment_value = sum(action[i] * current_prices[i] for i in range(len(action)))

            # Update portfolio value based on the new investments
            current_portfolio_value = current_investment_value

            # Append the updated portfolio value to the list
            portfolio_values.append(current_portfolio_value)
        else:
            print(f"Warning: current_step {env.current_step} is out of bounds. Skipping this step.")
            done = True
    
    if portfolio_values:
        return_metrics = ReturnMetrics(portfolio_values)
        daily_returns = return_metrics.daily_returns()

        file_path = f"./src/ibovespa/logs/cumulative_{period}_2024-10-07.csv"

        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "a", newline='') as file:
            writer = csv.writer(file)

            for i, daily_return in enumerate(daily_returns):
                writer.writerow([i, daily_return])
 
    file_path = f"./src/ibovespa/logs/testing_{period}_2024-10-07.txt"
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    for step, stocks in enumerate(selected_stocks):
        print(f"Step {step+1}: {stocks}")

        with open(file_path, "a") as file:
            file.write(f"Step {step+1}: {stocks}\n")

    env.print_action_history(period)
    print("Total reward:", rewards)

    with open(file_path, "a") as file:
        file.write(f"Total reward: {rewards}\n")
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    profit_criteria = ProfitCriteria(portfolio_values=portfolio_values, num_years=1)
    risk_criteria = RiskCriteria(returns=returns, portfolio_values=portfolio_values)
    risk_return_criteria = RiskReturnCriteria(returns=returns, portfolio_values=portfolio_values, risk_free_rate=0.01)

    print("Profit Criteria:")
    arr = profit_criteria.anualized_return()
    print(f"Anualized return: {arr}")

    print("Risk Criteria:")
    avol = risk_criteria.annualized_volatility()
    mdd = risk_criteria.max_drawdown()
    print(f"Annualized volatility: {avol}")
    print(f"Max drawdown: {mdd}")

    print("Risk-Return Criteria:")
    sharpe_ratio = risk_return_criteria.sharpe_ratio()
    calmar_ratio = risk_return_criteria.calculate_calmar_ratio()
    sortino_ratio = risk_return_criteria.calculate_sortino_ratio()
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Calmar Ratio: {calmar_ratio}")
    print(f"Sortino Ratio: {sortino_ratio}")

    file_path_metrics = "./src/ibovespa/logs/metrics_ibovespa_2024-10-07.csv"

    if not os.path.exists(file_path_metrics):
        os.makedirs(os.path.dirname(file_path_metrics), exist_ok=True)

    with open(file_path_metrics, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([period, arr, avol, mdd, sharpe_ratio, calmar_ratio, sortino_ratio])

    if i == 9:
        send_email(period)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, required=True, help="pre-pandemic, pandemic, post-pandemic, or all")
    args = parser.parse_args()

    for i in range(10):
        main(args.period, i)
