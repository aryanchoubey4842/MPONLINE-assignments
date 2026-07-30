import time
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # Enable rendering mode for visual output
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Load the trained model
    model = PPO.load("models/ppo_cartpole")
    
    for episode in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use deterministic actions for testing performance
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
            
            # Cap the frame rate slightly so it doesn't run too fast
            time.sleep(0.02)
            
        print(f"Episode {episode + 1} - Score: {total_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    main()
