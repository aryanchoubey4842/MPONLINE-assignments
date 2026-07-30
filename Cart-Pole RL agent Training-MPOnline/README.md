# Cart-Pole Balancing using Proximal Policy Optimization (PPO)

A Deep Reinforcement Learning project focused on training an autonomous agent to balance a pole on a moving cart using the CartPole-v1 physics simulator.

---

## Project Objective
The goal of this implementation is to leverage the Actor-Critic framework via PPO to master the balancing dynamics of the CartPole environment. Instead of utilizing hand-crafted rule-based control equations, the agent organically derives an optimal balancing policy through active exploration and cumulative reward maximization.

### Environment Specifications
* **Environment Name:** `CartPole-v1`
* **Observation Space:** 4-Dimensional continuous vector tracking cart position ($x$), cart velocity ($v_x$), pole angle ($\theta$), and pole angular velocity ($\dot{\theta}$).
* **Action Space:** Discrete (2) representing: Push cart left, or push cart right.
* **Reward:** +1 for every timestep the pole remains upright.
* **Termination:** Episode ends when the pole angle exceeds ±12° or cart moves beyond ±2.4 units, or after 500 timesteps (max score).

---

## Project Directory Layout
```text
Cart-Pole-PPO/
├── models/
│   └── ppo_cartpole.zip        # Saved trained PPO policy network weights
├── logs/
│   └── training_monitor.csv    # Evaluated step-by-step rolling environment metrics
├── videos/
│   └── rl-video-episode-0.mp4  # Recorded evaluation episode
├── graphs/
│   └── learning_curve.png      # Rendered performance progression plot
├── train.py                    # Policy initialization and baseline training loop
├── evaluate.py                 # Quantitative deterministic verification script
├── test.py                     # Visual environment deployment runtime execution
├── plot_training.py            # Custom parsing utility for monitoring logs
├── record_video.py             # Automated MP4 video export script
├── app.py                      # Flask web app for simulation UI
├── templates/
│   └── index.html              # Web interface template
├── static/
│   └── css/
│       └── style.css           # Web interface styling
└── requirements.txt            # System dependencies manifest
```

---

## System Setup & Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Cart-Pole-PPO
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

3. **Install the required package dependencies:**
```bash
pip install -r requirements.txt
```

---

## Hyperparameter Configuration

The implementation utilizes baseline settings optimized for multi-layer perceptron processing layers:

* **Network Policy:** `MlpPolicy`
* **Learning Rate ($\alpha$):** $3 \times 10^{-4}$
* **Discount Factor ($\gamma$):** $0.99$
* **Batch Size:** $64$
* **Rollout Steps ($n$):** $2048$
* **Optimization Epochs:** $10$
* **Clip Range:** $0.2$
* **Total Timesteps:** $100,000$

---

## Execution Guide

### 1. Training the Agent

Run the base optimization loop to generate experience trajectories up to 100,000 timesteps:

```bash
python train.py
```

### 2. Quantitative Evaluation

Run the model over multiple consecutive episodes to extract performance metrics:

```bash
python evaluate.py
```

### 3. Visual Human Rendering

Launch a real-time simulator window showing the trained weights controlling the cart:

```bash
python test.py
```

### 4. Automated Video Export

Generate `.mp4` recordings of completed balancing runs using the wrapper interface:

```bash
python record_video.py
```

### 5. Learning Curve Plotting

Generate the localized graph parsing the training history logs:

```bash
python plot_training.py
```

### 6. Web Application

Launch the Flask web interface for interactive simulation control:

```bash
python app.py
```

---

## Expected Performance

CartPole-v1 has a maximum possible score of **500** (the episode length cap). A well-trained PPO agent should consistently achieve near-perfect scores:

* **Target Mean Evaluation Score (20 Episodes):** $\geq 475$
* **Maximum Possible Score:** $500$

Achieving consistent scores near 500 confirms the agent has learned a stable, optimal pole-balancing policy.
