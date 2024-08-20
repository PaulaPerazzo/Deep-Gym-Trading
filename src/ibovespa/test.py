import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import torch
import pandas as pd
from gym_trading.agent import Agent
from gym_trading.enviroment import PortfolioEnv
from gym_trading.network import ActorCritic
from metrics.criteria import ProfitCriteria, RiskCriteria, RiskReturnCriteria
import argparse
from ibovespa.mailer import send_email

def main(period):
    stock_data = pd.read_csv("./data/ibovespa_2019_data.csv")
    
    # Configuração da rede
    actor_critic = ActorCritic(inputs_features=2850, n_actions=285, hidden_size=256)
    print("ActorCritic initialized.")
    
    if period == "pre-pandemic":
        # load data
        stock_data = pd.read_csv("./data/ibovespa_2019_data.csv")
        
        # load trained model
        model_path = "./models/ibovespa_pre_pandemic.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")

    elif period == "pandemic":
        stock_data = pd.read_csv("./data/ibovespa_2021_data.csv")

        model_path = "./models/ibovespa_pandemic.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")

    elif period == "post-pandemic":
        stock_data = pd.read_csv("./data/ibovespa_2023_data.csv")

        model_path = "./models/ibovespa_post_pandemic.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")
        
    elif period == "all":
        stock_data = pd.read_csv("./data/ibovespa_2023_data.csv")
        
        model_path = "./models/actor_critic_oficial_2.pth"
        actor_critic.load_state_dict(torch.load(model_path))
        actor_critic.eval()
        print("Model loaded.")

    stock_data = stock_data.set_index("Ticker")
    index_data = stock_data.copy()
    
    index_data = pd.DataFrame(index_data["^BVSP.4"])
    stock_data = stock_data.drop(columns=["^BVSP.4"])

    min_length = min(len(stock_data), len(index_data))
    stock_data = stock_data[-min_length:]
    index_data = index_data[-min_length:]

    # Inicializar o ambiente
    env = PortfolioEnv(stock_data, index_data)
    print("Environment initialized.")

    # Inicializar o agente
    agent = Agent(env, actor_critic, optimizer=None, scheduler=None)
    print("Agent initialized.")

    # Loop de avaliação
    state = env.reset().flatten() 
    done = False
    actions_taken = []
    rewards = []
    selected_stocks = []
    portfolio_values = []

    initial_portfolio_value = 100000  # Example initial value in cash
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
            current_portfolio_value = current_investment_value + current_portfolio_value

            # Append the updated portfolio value to the list
            portfolio_values.append(current_portfolio_value)
        else:
            print(f"Warning: current_step {env.current_step} is out of bounds. Skipping this step.")
            done = True
 
    file_path = f"./src/ibovespa/logs/testing_{period}.txt"
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Imprimir ou processar as ações e recompensas
    for step, stocks in enumerate(selected_stocks):
        print(f"Step {step+1}: {stocks}")

        with open(file_path, "a") as file:
            file.write(f"Step {step+1}: {stocks}\n")

    env.print_action_history(period)
    print("Total reward:", rewards)

    with open(file_path, "a") as file:
        file.write(f"Total reward: {rewards}\n")
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    profit_criteria = ProfitCriteria(portfolio_values=portfolio_values, total_steps=len(portfolio_values))
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

    file_path_metrics = f"./src/ibovespa/logs/metrics_{period}.txt"
    if not os.path.exists(file_path_metrics):
        os.makedirs(os.path.dirname(file_path_metrics), exist_ok=True)

    with open(file_path_metrics, "a") as file:
        file.write("Profit Criteria:\n")
        file.write(f"Anualized return: {arr}\n")
        file.write("Risk Criteria:\n")
        file.write(f"Annualized volatility: {avol}\n")
        file.write(f"Max drawdown: {mdd}\n")
        file.write("Risk-Return Criteria:\n")
        file.write(f"Sharpe Ratio: {sharpe_ratio}\n")
        file.write(f"Calmar Ratio: {calmar_ratio}\n")
        file.write(f"Sortino Ratio: {sortino_ratio}\n")

    send_email()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, required=True, help="pre-pandemic, pandemic, post-pandemic, or all")
    args = parser.parse_args()

    main(args.period)
