import torch
import pandas as pd
from gym_trading.agent import Agent
from gym_trading.enviroment import PortfolioEnv
from gym_trading.network import ActorCritic

def main():
    # Carregar os dados
    stock_data = pd.read_csv("../data/train_data_cleaned_nan.csv")
    stock_data.drop(columns=["Unnamed: 0"], inplace=True)
    stock_data = stock_data.sample(frac=0.05, random_state=42)
    stock_data = stock_data.set_index("Ticker")
    index_data = stock_data.copy()
    
    index_data = pd.DataFrame(index_data["^BVSP.4"])
    stock_data = stock_data.drop(columns=["^BVSP.4"])

    min_length = min(len(stock_data), len(index_data))
    stock_data = stock_data[-min_length:]
    index_data = index_data[-min_length:]

    # Inicializar o ambiente
    env = PortfolioEnv(stock_data, index_data)

    # Configuração da rede
    actor_critic = ActorCritic(inputs_features=2850, n_actions=285, hidden_size=256)

    # Carregar o modelo treinado
    model_path = "../models/actor_critic.pth"
    actor_critic.load_state_dict(torch.load(model_path))
    actor_critic.eval()

    # Inicializar o agente
    agent = Agent(env, actor_critic, optimizer=None, scheduler=None)

    # Loop de avaliação
    state = env.reset().flatten() 
    done = False
    actions_taken = []
    rewards = []
    selected_stocks = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = agent.decide_action(state_tensor, max_share=30).numpy().flatten()
        state, reward, done, _ = env.step(action)  
        # state, reward, done, _ = env.step(action.numpy())  
        state = state.flatten()  
        actions_taken.append(action)
        rewards.append(reward)

        stock_names = stock_data.columns
        selected = [stock_names[i] for i, num in enumerate(action) if num > 0]
        selected_stocks.append(selected)

    # Imprimir ou processar as ações e recompensas
    # print("Actions taken:", actions_taken)
    # print("Rewards:", rewards)

    for step, stocks in enumerate(selected_stocks):
        print(f"Step {step+1}: {stocks}")
    
    print("Total reward:", rewards)

if __name__ == "__main__":
    main()
