import os
import json
import argparse
import gym
import pybullet_envs
from stable_baselines3 import PPO, SAC

def load_json(path):
    # 인코딩 명시: 'utf-8'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(config_path):
    # 1) config 로드 (예: config_ppo_ant.json)
    config = load_json(config_path)
    algorithm = config["algorithm"]  # PPO 또는 SAC
    env_name = config["env_name"]    # AntBulletEnv-v0 등
    seed = config["seed"]
    timesteps = config["timesteps"]
    model_dir = config["model_dir"]
    hyperparams_path = config["hyperparams_path"]

    # 2) 알고리즘별 하이퍼파라미터 로드 (예: ppo_default.json, sac_default.json)
    hyperparams = load_json(hyperparams_path)

    # 3) 환경 생성
    env = gym.make(env_name)
    env.render(mode="human")
    env.seed(seed)

    # 4) 알고리즘 선택 후 모델 생성
    if algorithm == "PPO":
        model = PPO(
            policy=hyperparams["policy_type"],
            env=env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            gamma=hyperparams["gamma"],
            verbose=hyperparams["verbose"],
            tensorboard_log=hyperparams["tensorboard_log"]
        )
    elif algorithm == "SAC":
        model = SAC(
            policy=hyperparams["policy_type"],
            env=env,
            learning_rate=hyperparams["learning_rate"],
            buffer_size=hyperparams["buffer_size"],
            batch_size=hyperparams["batch_size"],
            tau=hyperparams["tau"],
            gamma=hyperparams["gamma"],
            train_freq=hyperparams["train_freq"],
            gradient_steps=hyperparams["gradient_steps"],
            ent_coef=hyperparams["ent_coef"],
            verbose=hyperparams["verbose"],
            tensorboard_log=hyperparams["tensorboard_log"]
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # 5) 학습
    print(f"[INFO] Training {algorithm} on {env_name} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    # 6) 모델 저장
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{algorithm.lower()}_{env_name.split('BulletEnv')[0].lower()}")
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file.")
    args = parser.parse_args()

    main(args.config)


'''
실행 예시

python train_rl.py --config configs/config_ppo_ant.json
python train_rl.py --config configs/config_ppo_hopper.json
python train_rl.py --config configs/config_sac_ant.json
python train_rl.py --config configs/config_sac_hopper.json
'''