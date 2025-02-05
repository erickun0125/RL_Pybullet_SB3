import gym
import pybullet_envs

env_id = "HopperBulletEnv-v0"

try:
    env = gym.make(env_id)
    print(f"환경 {env_id}가 정상적으로 로드되었습니다.")
except Exception as e:
    print(f"환경을 로드하는 중 오류 발생: {e}")
