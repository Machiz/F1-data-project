import os
import sys
import numpy as np
import pandas as pd
import torch
import torch as th
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import explained_variance

# Add project src to path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from models.f1_pit_env import F1PitEnv

class CustomPPO(PPO):
    """
    Custom PPO class that inherits from stable_baselines3.PPO.
    Overrides the train() method to add a print statement for each completed optimization epoch.
    """
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            epoch_pg_losses = []
            epoch_value_losses = []
            epoch_losses = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Collect for epoch-level prints
                epoch_pg_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_losses.append(loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
                
            # Print epoch done with losses (requested by user)
            mean_pg = np.mean(epoch_pg_losses) if epoch_pg_losses else 0.0
            mean_v = np.mean(epoch_value_losses) if epoch_value_losses else 0.0
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            print(f"  [Epoch Info] Optimization epoch {epoch + 1}/{self.n_epochs} completed | Policy Loss: {mean_pg:.6f} | Value Loss: {mean_v:.6f} | Total Loss: {mean_loss:.6f}")

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

def evaluate_strategy(env, model=None, strategy="model", num_episodes=20):
    """
    Evaluates a specific pit stop decision strategy on the environment.
    Strategies: 'model' (trained PPO), 'random', 'never_pit', 'real' (historical strategy).
    """
    rewards = []
    positions = []
    pit_counts = []
    reg_violations = []
    pit_laps = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        
        # Load historical pit data for the 'real' baseline
        race = info["race_name"]
        drv = info["driver_number"]
        driver_df = env.df[(env.df["race_name"] == race) & (env.df["driver_number"] == drv)].sort_values("lap_number")
        
        # Create a dictionary of lap_number -> action from historical data
        # NOTE: In our dataset, compound_ord on is_pit_lap=1 already reflects the NEW compound
        real_actions = {}
        for _, row in driver_df.iterrows():
            lp = int(row["lap_number"])
            if float(row["is_pit_lap"]) == 1.0:
                real_actions[lp] = int(row["compound_ord"])
            else:
                real_actions[lp] = 0
                
        while not done:
            current_lap = env.lap_number
            if strategy == "model":
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            elif strategy == "random":
                action = int(env.action_space.sample())
            elif strategy == "never_pit":
                action = 0
            elif strategy == "real":
                action = int(real_actions.get(env.lap_number, 0))
                if action not in [0, 1, 2, 3]:
                    action = 0
                    
            obs, reward, terminated, truncated, info = env.step(action)
            if action in [1, 2, 3]:
                pit_laps.append(current_lap)
            ep_reward += reward
            done = terminated or truncated
            
        rewards.append(ep_reward)
        positions.append(info["position"])
        pit_counts.append(info["stint_number"] - 1)
        # Regulation violation: used less than 2 compounds
        reg_violations.append(1.0 if len(info["used_compounds"]) < 2 else 0.0)
        
    return np.mean(rewards), np.mean(positions), np.mean(pit_counts), np.mean(reg_violations), pit_laps

def main():
    print("="*70)
    print("F1 Pit Strategic Decision RL Engine - Training & Evaluation")
    print("="*70)
    
    # 1. Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected for training: {device.upper()}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Instantiate Environments
    print("\nInstantiating training and evaluation environments...")
    # Wrap in a lambda for make_vec_env
    env_fn = lambda: F1PitEnv()
    train_env = make_vec_env(env_fn, n_envs=8) # Multi-processing training (increased to 8 for speed)
    eval_env = F1PitEnv()
    
    # 3. Create Eval Callback
    features_dir = Path(__file__).resolve().parent.parent.parent / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(features_dir),
        log_path=str(features_dir / "logs"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    
    # 4. Initialize PPO Model with long-horizon hyperparameters (Gamma = 0.99)
    print("\nInitializing Custom PPO agent for pit-window fine tuning...")
    best_model_path = features_dir / "best_model.zip"
    model_save_path = features_dir / "ppo_f1_pit_model.zip"
    fine_tuned_model_path = features_dir / "ppo_f1_pit_model_finetuned.zip"
    
    load_path = best_model_path if best_model_path.exists() else model_save_path
    if load_path.exists() and os.getenv("F1_PPO_RESET", "0") != "1":
        print(f"Loading existing PPO model for fine tuning: {load_path}")
        model = CustomPPO.load(
            str(load_path),
            env=train_env,
            device=device,
            custom_objects={
                "learning_rate": 8e-5,
                "lr_schedule": lambda _: 8e-5,
                "ent_coef": 0.0015,
                "gamma": 0.997,
                "gae_lambda": 0.96,
                "clip_range": lambda _: 0.15,
            },
        )
    else:
        model = CustomPPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=8e-5,
            n_steps=2048,
            batch_size=128,
            n_epochs=8,
            gamma=0.997,
            gae_lambda=0.96,
            clip_range=0.15,
            ent_coef=0.0015,
            verbose=1,
            device=device
        )
    
    # 5. Train PPO Agent
    total_timesteps = int(os.getenv("F1_PPO_TIMESTEPS", "200000"))
    
    print(f"\nFine tuning PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, reset_num_timesteps=False)
    model.save(str(fine_tuned_model_path))
    print(f"\n[OK] Fine-tuned model saved to: {fine_tuned_model_path}")
    
    # 7. Evaluate and Compare Strategies
    print("\n" + "="*50)
    print("EVALUATION & STRATEGY COMPARISON")
    print("="*50)
    
    strategies = ["never_pit", "random", "real", "model"]
    results = {}
    
    # Load the best model saved by callback
    if fine_tuned_model_path.exists():
        print(f"Loading fine-tuned model from: {fine_tuned_model_path}")
        best_model = CustomPPO.load(str(fine_tuned_model_path), env=eval_env)
    elif best_model_path.exists():
        print(f"Loading best evaluation model from: {best_model_path}")
        best_model = CustomPPO.load(str(best_model_path), env=eval_env)
    else:
        best_model = model
        
    for strat in strategies:
        print(f"Evaluating strategy: {strat.upper()}...")
        mean_reward, mean_pos, mean_pits, mean_violations, pit_lap_samples = evaluate_strategy(
            eval_env, model=best_model, strategy=strat, num_episodes=50
        )
        results[strat] = {
            "mean_reward": mean_reward,
            "mean_position": mean_pos,
            "mean_pits": mean_pits,
            "violations_pct": mean_violations * 100.0,
            "pit_laps": pit_lap_samples
        }
        
    # Print results summary table
    print("\n" + "-"*80)
    print(f"{'Strategy':<15} | {'Mean Reward':<12} | {'Mean Position':<14} | {'Mean Pit Stops':<15} | {'Reg Violations %':<15}")
    print("-"*80)
    for strat, res in results.items():
        print(f"{strat.upper():<15} | {res['mean_reward']:<12.2f} | {res['mean_position']:<14.2f} | {res['mean_pits']:<15.2f} | {res['violations_pct']:<15.2f}%")
    print("-"*80)
    
    model_pits = results.get("model", {}).get("pit_laps", [])
    if model_pits:
        print(f"MODEL pit-lap distribution: median={np.median(model_pits):.1f}, p25={np.percentile(model_pits, 25):.1f}, p75={np.percentile(model_pits, 75):.1f}, samples={len(model_pits)}")
    
    # 8. Plot and Save Learning Curves
    plot_learning_curve(features_dir)

def plot_learning_curve(features_dir):
    try:
        import matplotlib.pyplot as plt
        npz_path = features_dir / "logs" / "evaluations.npz"
        if not npz_path.exists():
            print(f"[WARNING] Evaluations log not found at: {npz_path}. Skipping plot.")
            return
            
        data = np.load(npz_path)
        timesteps = data["timesteps"]
        results = data["results"]
        
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, mean_rewards, label="PPO (Optuna-Tuned)", color="#1f77b4", linewidth=2)
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color="#1f77b4", alpha=0.2)
        
        plt.title("F1 Pit Strategic RL - PPO Learning Curve", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Training Timesteps (Laps)", fontsize=12)
        plt.ylabel("Evaluation Cumulative Reward", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        
        reports_dir = Path("c:/Users/marce/F1-data-project/project/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        plot_path = reports_dir / "learning_curve.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[OK] Learning curve plot saved to: {plot_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate learning curve plot: {e}")

if __name__ == "__main__":
    main()
