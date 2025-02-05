import os
import json
import argparse
import time
import gym
import pybullet_envs
from stable_baselines3 import PPO, SAC

def load_json(path):
    # 인코딩 명시: 'utf-8'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(config_path, num_episodes, render):
    # 1) config 로드
    config = load_json(config_path)
    algorithm = config["algorithm"]
    env_name = config["env_name"]
    model_dir = config["model_dir"]

    # 2) 모델 불러올 경로 결정
    #    학습 때 저장했던 파일명 규칙: f"{algorithm.lower()}_{env_name.split('BulletEnv')[0].lower()}.zip"
    model_filename = f"{algorithm.lower()}_{env_name.split('BulletEnv')[0].lower()}.zip"
    model_path = os.path.join(model_dir, model_filename)

    # 3) 환경 생성
    env = gym.make(env_name)
    env.render(mode="human")

    # 4) 알고리즘별 모델 로드
    if algorithm == "PPO":
        model = PPO.load(model_path)
    elif algorithm == "SAC":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    print(f"[INFO] Loaded {algorithm} model from {model_path}.")

    # 5) 평가
    total_rewards = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if render:
                env.render(mode="human")
                time.sleep(1/120)  # 시각화 속도 조절

        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}/{num_episodes}: Reward = {episode_reward:.2f}")

    print(f"[RESULT] Mean Reward over {num_episodes} episodes: {sum(total_rewards)/num_episodes:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    args = parser.parse_args()

    main(args.config, args.num_episodes, args.render)


'''
실행 예시

python eval_rl.py --config configs/config_ppo_ant.json --num_episodes 3 --render
python eval_rl.py --config configs/config_sac_hopper.json
'''