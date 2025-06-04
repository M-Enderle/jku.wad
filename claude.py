import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, namedtuple, OrderedDict
import random
import json
import onnx
import onnxruntime
import imageio.v2 as imageio
from copy import deepcopy
from typing import Dict, Tuple, List
import os

# Doom environment imports (assuming these are available)
from doom_arena import VizdoomMPEnv
from doom_arena.reward import VizDoomReward
from vizdoom import ScreenFormat
from IPython.display import HTML
import base64

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR SPEED
# =============================================================================

# Environment Configuration - OPTIMIZED
USE_GRAYSCALE = True  # Changed to True for faster processing
SCREEN_WIDTH = 128    # Reduced from 192
SCREEN_HEIGHT = 128   # Reduced from 256
SCREEN_CHANNELS = 1 if USE_GRAYSCALE else 3  # 1 channel is much faster
N_STACK_FRAMES = 2
EPISODE_TIMEOUT = 500  # Reduced from 1000 for faster episodes
ACTION_SPACE = 7

PLAYER_CONFIG = {
    "n_stack_frames": N_STACK_FRAMES,
    "extra_state": ["depth"],
    "hud": "none",
    "crosshair": True,
    "screen_format": "ScreenFormat.GRAY8" if USE_GRAYSCALE else "ScreenFormat.CRCGCB"
}

# Training Hyperparameters - OPTIMIZED
GAMMA = 0.99
N_STEP = 3
EPISODES = 2000
BATCH_SIZE = 64        # Increased from 16 for better GPU utilization
REPLAY_BUFFER_SIZE = 5000  # Reduced from 10000 to save memory
MIN_EPISODES = 5       # Reduced from 10
LEARNING_RATE = 5e-4   # Increased for faster learning
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998  # Faster decay
TARGET_UPDATE_FREQ = 500  # More frequent updates
EVAL_FREQUENCY = 50    # More frequent evaluation
SAVE_FREQUENCY = 250   # More frequent saving

# Performance optimizations
UPDATE_FREQUENCY = 4   # Update every 4 steps instead of every step
GRADIENT_ACCUMULATION = 1  # For larger effective batch sizes

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# ENHANCED REWARD FUNCTION
# =============================================================================

class EnhancedReward(VizDoomReward):
    def __init__(self, num_players: int):
        super().__init__(num_players)
        self.movement_history = deque(maxlen=70)  # Track recent actions
        self.last_health = None
        self.episode_steps = 0
        
    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Enhanced reward function:
        - Frag bonus: +150 per kill
        - Hit reward: +3 per hit
        - Damage penalty: -0.3 per hit taken
        - Movement bonus: +1 for movement that leads to hits/frags
        - Survival penalty: -0.02 per step to encourage engagement
        """
        self._step += 1
        self.episode_steps += 1
        
        # Core rewards
        hits_landed = game_var["HITCOUNT"] - game_var_old["HITCOUNT"]
        hits_taken = game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"]
        frags = game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"]
        
        rwd_frag = 150.0 * frags
        rwd_hit = 3.0 * hits_landed
        rwd_damage = -0.3 * hits_taken
        
        # Movement encouragement (reward recent movement if it led to success)
        movement_bonus = 0.0
        if (hits_landed > 0 or frags > 0) and len(self.movement_history) > 0:
            # Check if there was recent movement (simplified)
            movement_bonus = 1.0
            
        # Small survival penalty to encourage active play
        survival_penalty = -0.02
        
        # Health-based reward (optional - encourage staying alive)
        health_reward = 0.0
        current_health = game_var.get("HEALTH", 100)
        if self.last_health is not None:
            health_change = current_health - self.last_health
            if health_change > 0:  # Health gained (medkit, etc.)
                health_reward = 0.5
        self.last_health = current_health
        
        return rwd_hit, rwd_damage, rwd_frag, movement_bonus, survival_penalty
    
    def reset_episode(self):
        """Call this at the start of each episode"""
        self.movement_history.clear()
        self.last_health = None
        self.episode_steps = 0

# =============================================================================
# LIGHTWEIGHT DQN - OPTIMIZED FOR SPEED
# =============================================================================

class LightweightDQN(nn.Module):
    def __init__(self, input_channels: int, action_space: int, hidden: int = 256):
        super().__init__()
        
        # Simplified CNN encoder - much lighter
        self.encoder = nn.Sequential(
            # Block 1 - Aggressive downsampling
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Block 3 - Final feature extraction
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling for consistent output
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.feature_size = 64 * 4 * 4
        
        # Simple dueling streams - much lighter
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_space)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Normalize input to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

# =============================================================================
# SIMPLE REPLAY BUFFER - OPTIMIZED FOR SPEED
# =============================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SimpleReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def add(self, experience: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# N-STEP DQN AGENT
# =============================================================================

class NStepDQNAgent:
    def __init__(
        self,
        input_channels: int,
        action_space: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        device: torch.device = None
    ):
        self.action_space = action_space
        self.gamma = gamma
        self.n_step = n_step
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        # Networks
        self.q_network = LightweightDQN(input_channels, action_space).to(self.device)
        self.target_network = LightweightDQN(input_channels, action_space).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Replay buffer
        self.replay_buffer = SimpleReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        
        # Metrics
        self.training_step = 0
        
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        
        with torch.no_grad():
            # Fix state shape if needed
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            else:
                state = state.to(self.device)
            
            # Handle environment shape mismatches: [4, 2, H, W] -> [4, H, W]
            if len(state.shape) == 4 and state.shape[1] == 2:
                state = state[:, 0, :, :]  # Take first of the 2 channels
            elif len(state.shape) == 5:  # [1, 4, 2, H, W]
                state = state.squeeze(0)[:, 0, :, :]  # [4, H, W]
            elif len(state.shape) == 4 and state.shape[1] > 4:  # [4, channels, H, W]
                state = state[:, :4, :, :]  # Take first 4 channels
                if state.shape[1] == 2:
                    state = state[:, 0, :, :]  # [4, H, W]
            elif len(state.shape) == 3:  # Already correct [4, H, W]
                pass  # Use as is
            
            state = state.unsqueeze(0)  # Add batch dimension: [1, 4, H, W]
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        # Convert to tensors if needed and keep on CPU for storage
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device='cpu')
        else:
            state = state.cpu()  # Move to CPU for storage
            
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device='cpu')
        else:
            next_state = next_state.cpu()  # Move to CPU for storage
            
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Store experience when buffer is full
        if len(self.n_step_buffer) == self.n_step:
            n_step_return = self._compute_n_step_return()
            first_state, first_action = self.n_step_buffer[0][:2]
            last_next_state, last_done = self.n_step_buffer[-1][3:]
            
            experience = Experience(first_state, first_action, n_step_return, 
                                  last_next_state, last_done)
            self.replay_buffer.add(experience)
    
    def _compute_n_step_return(self) -> float:
        """Compute n-step discounted return"""
        n_step_return = 0
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * reward
            if done:
                break
        return n_step_return
    
    def update(self, batch_size: int = 32):
        if len(self.replay_buffer) < batch_size:
            return None
            
        # Only update every UPDATE_FREQUENCY steps for efficiency
        if self.training_step % UPDATE_FREQUENCY != 0:
            self.training_step += 1
            return None
            
        experiences = self.replay_buffer.sample(batch_size)
        if experiences is None:
            return None
        
        # Process states to handle environment shape mismatches
        processed_states = []
        processed_next_states = []
        
        for exp in experiences:
            state = exp.state.to(self.device)
            next_state = exp.next_state.to(self.device)
            
            # Handle shape: [4, 2, H, W] -> [4, H, W] by taking first channel of the extra dimension
            if len(state.shape) == 4 and state.shape[1] == 2:
                state = state[:, 0, :, :]  # Take first of the 2 channels: [4, H, W]
            elif len(state.shape) == 5:  # [1, 4, 2, H, W]
                state = state.squeeze(0)[:, 0, :, :]  # [4, H, W]
            elif len(state.shape) == 4 and state.shape[1] > 4:  # [4, channels, H, W]
                state = state[:, :4, :, :]  # Take first 4 channels
                if state.shape[1] == 2:
                    state = state[:, 0, :, :]  # [4, H, W]
            
            if len(next_state.shape) == 4 and next_state.shape[1] == 2:
                next_state = next_state[:, 0, :, :]
            elif len(next_state.shape) == 5:
                next_state = next_state.squeeze(0)[:, 0, :, :]
            elif len(next_state.shape) == 4 and next_state.shape[1] > 4:
                next_state = next_state[:, :4, :, :]
                if next_state.shape[1] == 2:
                    next_state = next_state[:, 0, :, :]
            
            processed_states.append(state)
            processed_next_states.append(next_state)
        
        states = torch.stack(processed_states)
        next_states = torch.stack(processed_next_states)
        
        # Debug print for the first update
        if self.training_step == 0:
            print(f"Debug: Original state shape: {experiences[0].state.shape}")
            print(f"Debug: Processed states shape: {states.shape}")
        
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values * (1 - dones)
        
        # Simple MSE loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)  # Reduced clip norm
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Learning rate schedule (less frequent)
        if self.training_step % 500 == 0:
            self.scheduler.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_checkpoint(self, filepath: str, episode: int, episode_rewards: list):
        """Save training checkpoint"""
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': episode_rewards,
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        return checkpoint['episode'], checkpoint['episode_rewards']

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def evaluate_agent(agent: NStepDQNAgent, env, num_episodes: int = 5) -> Dict[str, float]:
    """Evaluate agent performance"""
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Ensure state is on the correct device
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=agent.device)
            else:
                state = state.to(agent.device)
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state[0]
            episode_reward += reward[0]
            episode_length += 1
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards)
    }

def plot_training_progress(episode_rewards: list, eval_rewards: list, losses: list):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Training Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Smoothed episode rewards
    if len(episode_rewards) > 10:
        smoothed = pd.Series(episode_rewards).rolling(window=50, min_periods=1).mean()
        axes[0, 1].plot(smoothed)
        axes[0, 1].set_title('Smoothed Training Rewards (window=50)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
    
    # Evaluation rewards
    if eval_rewards:
        eval_episodes = np.arange(0, len(episode_rewards), len(episode_rewards) // len(eval_rewards))[:len(eval_rewards)]
        axes[1, 0].plot(eval_episodes, eval_rewards, 'ro-')
        axes[1, 0].set_title('Evaluation Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Eval Reward')
        axes[1, 0].grid(True)
    
    # Training loss
    if losses:
        axes[1, 1].plot(losses)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()

def train_agent(
    agent: NStepDQNAgent,
    env,
    reward_fn: EnhancedReward,
    num_episodes: int = 2000,
    eval_frequency: int = 50,  # More frequent evaluation
    target_update_frequency: int = 500,  # More frequent updates
    save_frequency: int = 250,  # More frequent saving
    verbose: bool = True
):
    """Optimized training loop"""
    episode_rewards = []
    eval_rewards = []
    losses = []
    best_eval_reward = float('-inf')
    best_model_state = None
    
    print(f"Starting optimized training for {num_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Replay buffer capacity: {agent.replay_buffer.capacity}")
    print(f"Update frequency: every {UPDATE_FREQUENCY} steps")
    print(f"Batch size: {BATCH_SIZE}")
    
    for episode in range(num_episodes):
        # Reset environment and reward function
        state = env.reset()[0]
        reward_fn.reset_episode()
        
        episode_reward = 0
        step_count = 0
        done = False
        
        # Episode rollout
        while not done and step_count < EPISODE_TIMEOUT:
            # Ensure state is on correct device
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=agent.device)
            else:
                state = state.to(agent.device)
                
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward[0], next_state[0], done)
            
            # Update agent more frequently but with update frequency control
            if episode > MIN_EPISODES and len(agent.replay_buffer) >= BATCH_SIZE:
                loss = agent.update(BATCH_SIZE)
                if loss is not None:
                    losses.append(loss)
            
            state = next_state[0]
            episode_reward += reward[0]
            step_count += 1
            
        episode_rewards.append(episode_reward)
        
        # Update target network
        if episode % target_update_frequency == 0 and episode > 0:
            agent.update_target_network()
            if verbose:
                print(f"Target network updated at episode {episode}")
        
        # Evaluation
        if episode % eval_frequency == 0 and episode > 0:
            eval_stats = evaluate_agent(agent, env, num_episodes=3)
            eval_reward = eval_stats['mean_reward']
            eval_rewards.append(eval_reward)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_model_state = agent.q_network.state_dict().copy()
                torch.save(best_model_state, f'best_model_ep_{episode}.pth')
            
            if verbose:
                print(f"Episode {episode:4d} | "
                      f"Train: {episode_reward:6.1f} | "
                      f"Eval: {eval_reward:6.1f} (±{eval_stats['std_reward']:.1f}) | "
                      f"Best: {best_eval_reward:6.1f} | "
                      f"ε: {agent.epsilon:.4f} | "
                      f"Steps: {step_count} | "
                      f"ReplayBuf: {len(agent.replay_buffer)} | "
                      f"Loss: {losses[-1] if losses else 'N/A'} | "
                      f"Recent avg: {np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward:6.1f}")
        
        # Save checkpoint
        if episode % save_frequency == 0 and episode > 0:
            agent.save_checkpoint(f'checkpoint_ep_{episode}.pth', episode, episode_rewards)
        
        # Simple progress display
        if episode % 10 == 0:
            recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
            print(f"Episode {episode:4d} | Recent avg: {recent_reward:6.1f} | ε: {agent.epsilon:.4f} | Buffer: {len(agent.replay_buffer)}")
        
        # Minimal memory cleanup (less frequent)
        if episode % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final evaluation and save
    print("\nTraining completed!")
    final_eval = evaluate_agent(agent, env, num_episodes=10)
    print(f"Final evaluation over 10 episodes:")
    print(f"  Mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  Max reward: {final_eval['max_reward']:.2f}")
    print(f"  Min reward: {final_eval['min_reward']:.2f}")
    
    # Load best model if available
    if best_model_state is not None:
        agent.q_network.load_state_dict(best_model_state)
        print(f"Loaded best model (eval reward: {best_eval_reward:.2f})")
    
    return episode_rewards, eval_rewards, losses, agent

# =============================================================================
# ENVIRONMENT AND AGENT INITIALIZATION - OPTIMIZED
# =============================================================================

reward_fn = EnhancedReward(num_players=1)
env = VizdoomMPEnv(
    num_players=1,
    num_bots=4,
    bot_skill=0,
    doom_map="ROOM",  # Simple, small map for faster training
    extra_state=PLAYER_CONFIG["extra_state"],
    episode_timeout=EPISODE_TIMEOUT,
    n_stack_frames=PLAYER_CONFIG["n_stack_frames"],
    crosshair=PLAYER_CONFIG["crosshair"],
    hud=PLAYER_CONFIG["hud"],
    reward_fn=reward_fn,
)

print(f"Observation space shape: {env.observation_space.shape}")
print(f"Using {'GRAYSCALE' if USE_GRAYSCALE else 'RGB'} mode")
print(f"Screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

agent = NStepDQNAgent(
    input_channels=4,  # Environment provides 4 stacked frames
    action_space=env.action_space.n,
    lr=LEARNING_RATE,
    gamma=GAMMA,
    n_step=N_STEP,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    device=device
)

print(f"Agent input channels: 4 (matching environment)")
print(f"Environment observation shape: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n}")
print(f"Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")

# Start optimized training
print("\n" + "="*60)
print("STARTING OPTIMIZED TRAINING")
print("="*60)

train_agent(agent, env, reward_fn, num_episodes=EPISODES)