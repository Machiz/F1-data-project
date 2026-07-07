import os
import sys
import gymnasium as gym
import optuna
import joblib
import numpy as np
import torch
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env

# Add project src to path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from models.f1_pit_env import F1PitEnv
from models.train_ppo_rl import CustomPPO

# Set optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    # 1. Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 5e-5, 1e-2, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.22)
    n_epochs = trial.suggest_int("n_epochs", 5, 12)
    
    # Ensure batch_size <= n_steps * n_envs
    n_envs = 4
    if batch_size > n_steps * n_envs:
        batch_size = n_steps * n_envs
    
    # 2. Instantiate envs
    env_fn = lambda: F1PitEnv()
    train_env = make_vec_env(env_fn, n_envs=n_envs)
    eval_env = F1PitEnv()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Create PPO Agent
    model = CustomPPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0,
        device=device
    )
    
    # 4. Train model (50k steps per trial for meaningful signal)
    model.learn(total_timesteps=50000)
    
    # 5. Evaluate agent performance
    eval_rewards = []
    eval_pits = []
    eval_violations = []
    eval_positions = []
    num_eval_episodes = 10
    
    for _ in range(num_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        eval_rewards.append(ep_reward)
        eval_pits.append(info["stint_number"] - 1)
        eval_violations.append(1.0 if len(info["used_compounds"]) < 2 else 0.0)
        eval_positions.append(info["position"])
        
    mean_reward = np.mean(eval_rewards)
    mean_pits = np.mean(eval_pits)
    violation_rate = np.mean(eval_violations)
    mean_position = np.mean(eval_positions)
    
    # Pit-window objective: reward is still the main performance signal, but
    # policies with more than two stops are demoted because they do not answer
    # the business question of choosing the optimal pit lap/window.
    extra_pit_penalty = max(0.0, mean_pits - 2.0) * 45.0
    violation_penalty = violation_rate * 250.0
    position_penalty = max(0.0, mean_position - 14.0) * 2.0
    objective_score = mean_reward - extra_pit_penalty - violation_penalty - position_penalty
    
    # Report intermediate values for pruning
    trial.set_user_attr("mean_pits", float(mean_pits))
    trial.set_user_attr("violation_rate", float(violation_rate))
    trial.set_user_attr("mean_reward", float(mean_reward))
    trial.set_user_attr("mean_position", float(mean_position))
    trial.set_user_attr("objective_score", float(objective_score))
    
    print(f"  Trial {trial.number}: score={objective_score:.2f}, reward={mean_reward:.2f}, pits={mean_pits:.1f}, pos={mean_position:.1f}, violations={violation_rate*100:.0f}%")
    
    return objective_score

def main():
    print("="*70)
    print("F1 Pit Strategic Decision RL Engine - Hyperparameter Tuning (Optuna)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for tuning: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create Optuna study to maximize mean evaluation reward
    study = optuna.create_study(direction="maximize")
    
    n_trials = 10
    print(f"\nRunning {n_trials} trials (50k steps each, ~2 min per trial)...")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "="*70)
    print("OPTUNA STUDY RESULTS")
    print("="*70)
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"Best Trial Reward: {study.best_value:.2f}")
    print(f"Best Trial Pits:   {study.best_trial.user_attrs.get('mean_pits', 'N/A')}")
    print(f"Best Trial Violations: {study.best_trial.user_attrs.get('violation_rate', 'N/A')}")
    print("\nBest Hyperparameters:")
    for param_name, param_val in study.best_params.items():
        print(f"  * {param_name}: {param_val}")
    print("="*70)
    
    # Save the best parameters to joblib
    features_dir = Path(__file__).resolve().parent.parent.parent / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = features_dir / "ppo_best_hyperparameters.joblib"
    joblib.dump(study.best_params, best_params_path)
    print(f"\nBest hyperparameters saved to: {best_params_path}")
    
    # Print all trials summary
    print("\n" + "-"*80)
    print(f"{'Trial':<8} | {'Reward':<10} | {'Pits':<8} | {'Violations':<12} | {'LR':<12} | {'Gamma':<8} | {'Ent Coef':<10}")
    print("-"*80)
    for t in study.trials:
        pits = t.user_attrs.get("mean_pits", "N/A")
        viol = t.user_attrs.get("violation_rate", "N/A")
        lr = t.params.get("learning_rate", "N/A")
        gamma = t.params.get("gamma", "N/A")
        ent = t.params.get("ent_coef", "N/A")
        val = t.value if t.value is not None else "FAILED"
        print(f"{t.number:<8} | {val:<10.2f} | {pits:<8} | {viol:<12} | {lr:<12.6f} | {gamma:<8.4f} | {ent:<10.6f}")
    print("-"*80)

if __name__ == "__main__":
    main()
