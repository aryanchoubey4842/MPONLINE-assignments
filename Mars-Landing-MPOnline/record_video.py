import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

def main():
    # Ensure the videos directory exists
    os.makedirs("videos", exist_ok=True)

    # Initialize environment in rgb_array mode with Mars gravity (-3.721 m/s^2)
    env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=-3.721)
    
    # Wrap the environment to trigger video recording for the episode
    env = RecordVideo(
        env, 
        video_folder="videos", 
        episode_trigger=lambda episode: True
    )
    
    # Load your trained model
    model = PPO.load("models/ppo_lunarlander")
    
    obs, info = env.reset()
    done = False
    
    print("--- Recording Episode Video ---")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    env.close()
    
    print("Applying Mars tint to video...")
    import glob
    from moviepy import VideoFileClip
    import numpy as np

    def mars_tint(image):
        mars_img = image.copy().astype(float)
        mars_img[:, :, 0] = np.clip(mars_img[:, :, 0] * 1.5, 0, 255) # boost red
        mars_img[:, :, 1] = np.clip(mars_img[:, :, 1] * 0.8, 0, 255) # lower green
        mars_img[:, :, 2] = np.clip(mars_img[:, :, 2] * 0.5, 0, 255) # lower blue
        return mars_img.astype(np.uint8)

    list_of_files = glob.glob('videos/*.mp4')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        if 'rl-video-episode' in latest_file:
            clip = VideoFileClip(latest_file)
            mars_clip = clip.image_transform(mars_tint)
            mars_file = latest_file.replace('.mp4', '-mars.mp4')
            mars_clip.write_videofile(mars_file, logger=None)
            clip.close()
            mars_clip.close()
            os.remove(latest_file)

    print("Video successfully saved in the videos/ folder.")

if __name__ == "__main__":
    main()