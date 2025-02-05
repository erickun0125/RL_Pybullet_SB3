# utils/utils_plot.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_monitor_csv(csv_path, window=50, title="Training Rewards"):
    """
    SB3 Monitor 래퍼의 monitor.csv (t, r, l) 포맷을 읽고,
    rolling average로 보상 곡선을 그림.
    """
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return

    # monitor.csv의 첫 줄(혹은 여러 줄)에 주석(#)이 있을 수 있으니 처리
    # columns: t, r, l
    df = pd.read_csv(csv_path, comment="#", header=None, names=["t", "r", "l"])

    # Rolling mean
    df["r_smooth"] = df["r"].rolling(window, min_periods=1).mean()

    plt.figure(figsize=(8,4))
    plt.plot(df["r"], alpha=0.3, label="Reward")
    plt.plot(df["r_smooth"], label=f"Reward (Rolling {window})", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_training_curves(algo="PPO", env="AntBulletEnv-v0", seed=42, root_log_dir="logs", window=50):
    """
    ex) logs/PPO_AntBulletEnv-v0_seed42/monitor.csv
    """
    run_name = f"{algo.upper()}_{env}_seed{seed}"
    csv_path = os.path.join(root_log_dir, run_name, "monitor.csv")
    plot_monitor_csv(csv_path, window=window, title=f"{run_name} Training Rewards")

if __name__ == "__main__":
    # 테스트 예시
    plot_training_curves(algo="PPO", env="AntBulletEnv-v0", seed=42)

'''
python -m utils.utils_plot
'''