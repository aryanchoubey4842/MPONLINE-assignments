import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    env = gym.make("CartPole-v1")
    
    # Load the trained weights
    model = PPO.load("models/ppo_cartpole")
    
    print("--- Running Evaluation over 20 Episodes ---")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True
    )
    
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()
