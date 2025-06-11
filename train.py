from typing import Dict, Sequence

import torch
from collections import deque, OrderedDict
from copy import deepcopy
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import vizdoom as vzd
from vizdoom import ScreenFormat
import logging  # Import the logging module
import os
import json
from datetime import datetime

from gym import Env
from torch import nn
from einops import rearrange

from doom_arena import VizdoomMPEnv
from doom_arena.reward import VizDoomReward
from doom_arena.render import render_episode
from IPython.display import HTML
from typing import Dict, Tuple
from rich.logging import RichHandler  # Add this import
from rich.table import Table
from rich.console import Console

# from doom_arena.reward import VizDoomReward # Already imported


USE_GRAYSCALE = False  # ← flip to False for RGB

PLAYER_CONFIG = {
    "n_stack_frames": 3,
    "extra_state": ["depth"],
    "hud": "none",
    "crosshair": True,
    "screen_format": 8 if USE_GRAYSCALE else 0,
}

# TODO: environment training paramters
N_STACK_FRAMES = 3
NUM_BOTS = 4
EPISODE_TIMEOUT = 2000
# TODO: model hyperparams
GAMMA = 0.995
EPISODES = 1000
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 15000
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1 / (EPISODES * 0.7)
N_EPOCHS = 20

# ================================================================
# Logging Setup
# ================================================================
LOG_FILENAME = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level

# Create a file handler (keep for file logging)
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setLevel(logging.INFO)

# Create a Rich console handler
console_handler = RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)
console_handler.setLevel(logging.INFO)

# Use default formatter for file, RichHandler handles its own formatting
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
# Do not set formatter for RichHandler

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to the root logger if it has handlers (e.g., in IPython/Jupyter)
logger.propagate = False

# Log all main parameters as a table at the beginning
if console_handler:
    table = Table(title="Training Parameters")
    table.add_column("Parameter", style="bold")
    table.add_column("Value")
    param_dict = {
        "USE_GRAYSCALE": USE_GRAYSCALE,
        "PLAYER_CONFIG": str(PLAYER_CONFIG),
        "N_STACK_FRAMES": N_STACK_FRAMES,
        "NUM_BOTS": NUM_BOTS,
        "EPISODE_TIMEOUT": EPISODE_TIMEOUT,
        "GAMMA": GAMMA,
        "EPISODES": EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "REPLAY_BUFFER_SIZE": REPLAY_BUFFER_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "EPSILON_START": EPSILON_START,
        "EPSILON_END": EPSILON_END,
        "EPSILON_DECAY": EPSILON_DECAY,
        "N_EPOCHS": N_EPOCHS,
    }
    for k, v in param_dict.items():
        table.add_row(str(k), str(v))
    # Use Rich's Console to print the table (not logger, as logger.info won't render tables)
    console = Console()
    console.print(table)

device = "cuda" if torch.cuda.is_available() else "cpu"  # Check for cuda availability
DTYPE = torch.float32

env = VizdoomMPEnv(
    num_players=1,
    num_bots=NUM_BOTS,
    bot_skill=0,
    doom_map="ROOM",  # NOTE simple, small map; other options: TRNM, TRNMBIG
    extra_state=PLAYER_CONFIG[
        "extra_state"
    ],  # see info about states at the beginning of 'Environment configuration' above
    episode_timeout=EPISODE_TIMEOUT,
    n_stack_frames=PLAYER_CONFIG["n_stack_frames"],
    crosshair=PLAYER_CONFIG["crosshair"],
    hud=PLAYER_CONFIG["hud"],
    screen_format=PLAYER_CONFIG["screen_format"],
    reward_fn=None,
)


# ================================================================
# DQN — design your network here
# ================================================================


class DQN(nn.Module):
    def __init__(self, input_dim: int, action_space: int, hidden: int = 256):
        super().__init__()

        # CNN encoder for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4, padding=2),  # input_dim is now C * N_STACK_FRAMES
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # MLP head to map features to action Q-values
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, hidden),  # For 128x128 input, 16x16 feature map
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_space),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        # Rearrange dimensions if input has extra state dimension
        if len(frame.shape) == 5:
            frame = rearrange(frame, 'b s c h w -> b (s c) h w')
        x = self.encoder(frame)  # Process frame through CNN encoder
        x = self.head(x)        # Map features to Q-values
        return x

# ================================================================
# Weighted Replay Buffer for prioritizing positive rewards
# ================================================================


class WeightedReplayBuffer:
    def __init__(self, capacity: int, positive_weight: float = 100.0):
        self.buffer = deque(maxlen=capacity)
        self.positive_weight = positive_weight

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        # Calculate weights based on rewards
        weights = []
        for exp in self.buffer:
            _, _, reward, _, _ = exp
            if reward > 0:
                weights.append(self.positive_weight)
            else:
                weights.append(1.0)

        # Normalize weights
        if not weights:  # Handle empty buffer case
            return []
        weights = np.array(weights)
        if (
            weights.sum() == 0
        ):  # Handle case where all weights are zero (e.g. if positive_weight is 0 and all rewards are <=0)
            weights = np.ones_like(weights) / len(weights)  # Uniform sampling
        else:
            weights = weights / weights.sum()

        # Sample with replacement based on weights
        indices = np.random.choice(
            len(self.buffer), size=batch_size, p=weights, replace=True
        )  # Added replace=True
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ================================================================
# Utility functions
# ================================================================


def epsilon_greedy(
    env_obj, model_obj, obs_obj, epsilon_val, device_obj, dtype_obj
):  # Renamed to avoid conflict
    """Epsilon-greedy action selection"""
    if random.random() < epsilon_val:
        return env_obj.action_space.sample()
    else:
        with torch.no_grad():
            obs_tensor = torch.tensor(
                obs_obj, device=device_obj, dtype=dtype_obj
            ).unsqueeze(0)
            q_values = model_obj(obs_tensor)
            return q_values.argmax().item()


def update_ema(target_model, main_model, tau=0.005):
    """Exponential moving average update for target network"""
    for target_param, main_param in zip(
        target_model.parameters(), main_model.parameters()
    ):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


def linear_epsilon_decay(
    episode_num, start_eps_val, end_eps_val, decay_episodes_val
):  # Renamed
    """Linear epsilon decay for more exploration"""
    if episode_num >= decay_episodes_val:
        return end_eps_val
    return start_eps_val - (start_eps_val - end_eps_val) * (
        episode_num / decay_episodes_val
    )


# ================================================================
# Initialise your networks and training utilities
# ================================================================
# main Q-network
in_channels = env.observation_space.shape[0] * N_STACK_FRAMES  # 1 if grayscale, else 3/4
model = DQN(
    input_dim=in_channels,
    action_space=env.action_space.n,
    hidden=256,  # increased hidden size
).to(device, dtype=DTYPE)

# Target network (hard copy)
model_tgt = deepcopy(model).to(
    device, dtype=DTYPE
)  # Ensure target is also on the correct device and dtype

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# Weighted replay buffer
replay_buffer = WeightedReplayBuffer(capacity=REPLAY_BUFFER_SIZE, positive_weight=100.0)
epsilon = EPSILON_START
epsilon_decay_episodes = int(0.7 * EPISODES)  # Decay over 70% of episodes



# ---------------------  TRAINING LOOP  ----------------------
reward_list, q_loss_list = [], []
best_eval_return, best_model_state = float("-inf"), None  # Store model state_dict

for episode in range(EPISODES):
    ep_metrics = {
        "custom_reward": 0.0,
        "hits": 0,
        "frags": 0,
        "hits_taken": 0,
        "deaths": 0,
        "kills": 0,
        "distance_moved": 0,
    }
    obs_tuple = env.reset()
    obs = obs_tuple[0]  # Assuming env.reset() returns a tuple (obs, info)
    done, ep_return = False, 0.0
    model.eval()  # Set model to evaluation mode for rollout

    # ───────── rollout ─────────────────────────────────────────────
    step_count = 0
    while not done:
        act = epsilon_greedy(env, model, obs, epsilon, device, DTYPE)
        next_obs_tuple, rwd_raw_tuple, done, info_tuple = env.step(
            act
        )  # Expecting tuples

        next_obs = next_obs_tuple[0]  # Actual observation

        # Enhanced reward definition using game variables
        current_game_env = (
            env.envs[0].unwrapped if hasattr(env.envs[0], "unwrapped") else env.envs[0]
        )
        gv = current_game_env._game_vars
        gv_pre = current_game_env._game_vars_pre

        hit_reward = 10.0 * (gv.get("HITCOUNT", 0) - gv_pre.get("HITCOUNT", 0))
        kill_reward = 100.0 * (gv.get("KILLCOUNT", 0) - gv_pre.get("KILLCOUNT", 0))
        hit_taken_penalty = -1.0 * (
            gv.get("HITS_TAKEN", 0) - gv_pre.get("HITS_TAKEN", 0)
        )
        frag_reward = 20.0 * (gv.get("FRAGCOUNT", 0) - gv_pre.get("FRAGCOUNT", 0))
        armor_reward = 10.0 * (gv.get("ARMOR", 0) - gv_pre.get("ARMOR", 0))
        death_penalty = -20.0 * (gv.get("DEATHCOUNT", 0) - gv_pre.get("DEATHCOUNT", 0))

        custom_rwd = (
            hit_reward
            + hit_taken_penalty
            + frag_reward
            + armor_reward
            + death_penalty
        )

        # Track episode metrics
        ep_metrics["custom_reward"] += custom_rwd
        ep_metrics["hits"] += gv.get("HITCOUNT", 0) - gv_pre.get("HITCOUNT", 0)
        ep_metrics["frags"] += gv.get("FRAGCOUNT", 0) - gv_pre.get("FRAGCOUNT", 0)
        ep_metrics["hits_taken"] += gv.get("HITS_TAKEN", 0) - gv_pre.get(
            "HITS_TAKEN", 0
        )
        ep_metrics["deaths"] += gv.get("DEATHCOUNT", 0) - gv_pre.get("DEATHCOUNT", 0)
        ep_metrics["kills"] += gv.get("KILLCOUNT", 0) - gv_pre.get("KILLCOUNT", 0)
        ep_metrics["deaths"] += gv.get("DEATHCOUNT", 0) - gv_pre.get("DEATHCOUNT", 0)
        ep_metrics["distance_moved"] += abs(
            gv.get("POSITION_X", 0) - gv_pre.get("POSITION_X", 0)
        ) + abs(gv.get("POSITION_Y", 0) - gv_pre.get("POSITION_Y", 0))

        replay_buffer.append(
            (obs, act, custom_rwd, next_obs, done)
        )  # Storing the actual next_obs
        obs, ep_return = next_obs, ep_return + custom_rwd
        step_count += 1
        if (
            EPISODE_TIMEOUT and step_count >= EPISODE_TIMEOUT
        ):  # Manual timeout check if env doesn't handle it internally for 'done'
            done = True

    reward_list.append(ep_return)

    # ───────── learning step (experience replay) ──────────────────
    if len(replay_buffer) >= BATCH_SIZE:
        model.train()  # Set model to training mode for learning
        epoch_losses = []

        for _ in range(N_EPOCHS):
            batch = replay_buffer.sample(BATCH_SIZE)
            if not batch:
                continue  # Skip if buffer sampling returns empty

            s_list, a_list, r_list, s2_list, d_list = zip(*batch)

            s = torch.stack(
                [torch.as_tensor(item, device=device, dtype=DTYPE) for item in s_list]
            )
            s2 = torch.stack(
                [torch.as_tensor(item, device=device, dtype=DTYPE) for item in s2_list]
            )
            a = torch.tensor(
                a_list, device=device, dtype=torch.long
            )  # Actions are indices
            r = torch.tensor(r_list, device=device, dtype=torch.float32)
            d = torch.tensor(d_list, device=device, dtype=torch.float32)

            # Current Q-values
            q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # Target Q-values (Double DQN style)
            with torch.no_grad():
                next_actions = model(s2).argmax(1)
                q2 = model_tgt(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                tgt = r + GAMMA * q2 * (1 - d)

            loss = F.mse_loss(q, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            # Soft update target network
            update_ema(model_tgt, model)

        if epoch_losses:  # Only extend if there were losses
            q_loss_list.extend(epoch_losses)

    # Update learning rate and epsilon
    scheduler.step()
    epsilon = linear_epsilon_decay(
        episode, EPSILON_START, EPSILON_END, epsilon_decay_episodes
    )

    # Log progress
    if (
        episode + 1
    ) % 10 == 0 or episode < 5:  # Log more frequently at the beginning and then every 10 episodes
        avg_loss = (
            np.mean(q_loss_list[-N_EPOCHS * BATCH_SIZE :])
            if q_loss_list
            else float("nan")
        )
        # Log all ep_metrics keys and values, formatting floats and ints nicely
        ep_metrics_str = " | ".join(
            f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in ep_metrics.items()
        )
        logger.info(
            f"Ep {episode+1:04}/{EPISODES} | Return: {ep_return:7.1f} | "
            f"{ep_metrics_str} | Steps: {step_count:4} | "
            f"ε: {epsilon:.3f} | Avg Loss (last epoch): {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    # ───────── evaluation for best-model tracking ─────────────────
    if (episode + 1) % 50 == 0:  # Evaluate every 50 episodes
        logger.info(f"Starting evaluation for episode {episode+1}...")

        eval_returns = []
        eval_metrics_runs = []
        for eval_run in range(3):
            eval_obs_tuple, eval_done, eval_return = env.reset(), False, 0.0
            eval_obs = eval_obs_tuple[0]
            model.eval()
            eval_steps = 0
            current_eval_metrics = {"hits": 0, "frags": 0, "hits_taken": 0}

            while not eval_done:
                act = epsilon_greedy(
                    env, model, eval_obs, 0.01, device, DTYPE
                )  # Small epsilon for evaluation
                eval_next_obs_tuple, eval_rwd_tuple, eval_done, eval_info_tuple = env.step(
                    act
                )

                eval_obs = eval_next_obs_tuple[0]

                current_eval_game_env = (
                    env.envs[0].unwrapped
                    if hasattr(env.envs[0], "unwrapped")
                    else env.envs[0]
                )
                eval_gv = current_eval_game_env._game_vars
                eval_gv_pre = current_eval_game_env._game_vars_pre

                eval_hit_rwd = 2.0 * (
                    eval_gv.get("HITCOUNT", 0) - eval_gv_pre.get("HITCOUNT", 0)
                )
                eval_ht_rwd = -0.1 * (
                    eval_gv.get("HITS_TAKEN", 0) - eval_gv_pre.get("HITS_TAKEN", 0)
                )
                eval_frag_rwd = 100.0 * (
                    eval_gv.get("FRAGCOUNT", 0) - eval_gv_pre.get("FRAGCOUNT", 0)
                )
                eval_current_rwd = eval_hit_rwd + eval_ht_rwd + eval_frag_rwd

                eval_return += eval_current_rwd  # Accumulate custom reward for evaluation

                current_eval_metrics["hits"] += eval_gv.get(
                    "HITCOUNT", 0
                ) - eval_gv_pre.get("HITCOUNT", 0)
                current_eval_metrics["frags"] += eval_gv.get(
                    "FRAGCOUNT", 0
                ) - eval_gv_pre.get("FRAGCOUNT", 0)
                current_eval_metrics["hits_taken"] += eval_gv.get(
                    "HITS_TAKEN", 0
                ) - eval_gv_pre.get("HITS_TAKEN", 0)

                eval_steps += 1
                if EPISODE_TIMEOUT and eval_steps >= EPISODE_TIMEOUT:
                    eval_done = True

            eval_returns.append(eval_return)
            eval_metrics_runs.append({
                "hits": current_eval_metrics["hits"],
                "frags": current_eval_metrics["frags"],
                "hits_taken": current_eval_metrics["hits_taken"],
                "steps": eval_steps,
            })

        avg_eval_return = np.mean(eval_returns)
        avg_eval_hits = np.mean([m["hits"] for m in eval_metrics_runs])
        avg_eval_frags = np.mean([m["frags"] for m in eval_metrics_runs])
        avg_eval_hits_taken = np.mean([m["hits_taken"] for m in eval_metrics_runs])
        avg_eval_steps = np.mean([m["steps"] for m in eval_metrics_runs])

        logger.info(
            f"Evaluation after Ep {episode+1}: "
            f"Avg Return: {avg_eval_return:7.1f} | "
            f"Avg Hits: {avg_eval_hits:3.1f} | "
            f"Avg Frags: {avg_eval_frags:2.1f} | "
            f"Avg Taken: {avg_eval_hits_taken:3.1f} | "
            f"Avg Steps: {avg_eval_steps:.1f} "
        )

        if avg_eval_return > best_eval_return:
            best_eval_return = avg_eval_return
            best_model_state = deepcopy(model.state_dict())  # Save state_dict
            logger.info(
                f"  → New best avg eval return: {best_eval_return:.1f}. Model saved."
            )

            # save to folder
            model_save_path = (
                f'models/dqn_doom_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}_'
                f'{episode+1}_{avg_eval_return:.1f}_{int(round(avg_eval_hits))}_{int(round(avg_eval_frags))}_{int(round(avg_eval_hits_taken))}.pth'
            )
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved to {model_save_path}")


# ---------------------  SAVE / EXPORT ---------------------------------------
logger.info("Training completed!")

final_model_state = (
    best_model_state if best_model_state is not None else model.state_dict()
)
model_save_path = f'dqn_doom_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'

save_dict = {
    "model_state_dict": final_model_state,
    "optimizer_state_dict": optimizer.state_dict(),
    "reward_list": reward_list,
    "q_loss_list": q_loss_list,
    "best_eval_return": best_eval_return,
    "final_epsilon": epsilon,
    "hyperparams": {
        "GAMMA": GAMMA,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "REPLAY_BUFFER_SIZE": REPLAY_BUFFER_SIZE,
        "EPSILON_START": EPSILON_START,
        "EPSILON_END": EPSILON_END,
        "N_EPOCHS": N_EPOCHS,
        "EPISODES": EPISODES,
        "NUM_BOTS": NUM_BOTS,
        "EPISODE_TIMEOUT": EPISODE_TIMEOUT,
        "PLAYER_CONFIG": PLAYER_CONFIG,
        "USE_GRAYSCALE": USE_GRAYSCALE,
    },
}

try:
    torch.save(save_dict, model_save_path)
    logger.info(f"Final model and training data saved to {model_save_path}")
except Exception as e:
    logger.error(f"Error saving model: {e}")


logger.info(f"Best evaluation return during training: {best_eval_return:.1f}")
if reward_list:
    logger.info(f"Final episode return: {reward_list[-1]:.1f}")
    logger.info(
        f"Average return (last 100 episodes): {np.mean(reward_list[-100:]):.1f}"
    )
else:
    logger.info("No episodes were completed to calculate final/average returns.")

# Close logging handlers
file_handler.close()
console_handler.close()
logger.removeHandler(file_handler)
logger.removeHandler(console_handler)
