import torch
import os

class Agent:
    def __init__(self, env, model, optimizer, scheduler, eps_min=0.1, eps=1.0, eps_decay=0.995):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_shares = 30
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eps_decay_strategy = "linear"
    

    def update_eps(self, episode):
        if self.eps_decay_strategy == "linear":
            if episode == 0:
                self.eps = 1.0
            else:
                self.eps = max(self.eps_min, self.eps - (self.eps - self.eps_min) / episode)
        elif self.eps_decay_strategy == "exponential":
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
        print(f"Updated epsilon: {self.eps}")

    
    def decide_action(self, state, max_share=30):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        state = state.reshape(1, -1)  # This will flatten [1, 10, 285] to [1, 2850]

        if torch.rand(1) < self.eps:
            state_shape_1 = (int(state.shape[1]) / 10)
            action = torch.randint(0, state.shape[1], size=(1, state_shape_1))
            # action = torch.randint(0, state.shape[1], size=(1, 285))
        
        else:
            with torch.no_grad():
                value, dist = self.model(state)
                probs = dist.probs.squeeze()
            
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("Probs contain NaN or inf values:", probs)
                probs = torch.nan_to_num(probs)
                
            action = self.sample_action(probs, max_share)

        return action


    def sample_action(self, probs, max_share):
        probs = torch.relu(probs)

        probs /= probs.sum()
        probs = torch.clamp(probs, min=1e-6)

        scaled_probs = (probs / probs.sum()) * max_share
        shares = torch.floor(scaled_probs)

        while shares.sum() < max_share:
            residuals = scaled_probs - shares
            residuals = torch.clamp(residuals, min=0)
            residuals /= residuals.sum()
            choosen_index = torch.multinomial(residuals, 1)
            shares[choosen_index] += 1
        
        return shares.int()

    
    def train(self, n_episodes):
        for episode in range(n_episodes):
            state = self.env.reset().flatten()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                value, dist = self.model(state_tensor)
                probs = dist.probs.squeeze()
                action = self.sample_action(probs, self.max_shares)

                next_state, reward, done, _ = self.env.step(action.numpy())
                next_state = next_state.flatten()
                total_reward += reward

                ### calculate the loss
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                next_value, _ = self.model(next_state_tensor)
                reward_tensor = torch.tensor(reward, dtype=torch.float32).squeeze()
                target = reward_tensor + (self.env.gamma * next_value)
                error = target - value
                critic_loss = error.pow(2)

                action_tensor = torch.tensor(action, dtype=torch.long)

                if action_tensor.dim() > 1:
                    action_tensor = action_tensor.squeeze()

                if len(probs) == 30:
                    actor_loss = -torch.log(probs[action_tensor - 1] + 1e-6) * error.detach()
                else:
                    actor_loss = -torch.log(probs[action_tensor] + 1e-6) * error.detach()

                loss = (actor_loss + critic_loss).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
            
            # update episilon after each episode
            self.update_eps(episode)
            self.scheduler.step()
            print(f"Episode {episode} - Reward: {total_reward}")

            with open("./src/gym_trading/logs/training_3.txt", "a") as f:
                f.write(f"Episode {episode} - Reward: {total_reward}\n")

        print("Training finished")
    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
