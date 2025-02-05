# scripts/callbacks.py

import os
import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    """
    일정 스텝마다 PyBullet 환경 화면을 캡처하여 이미지로 저장하는 콜백.
    """
    def __init__(self, render_freq=10000, save_dir="logs/screenshots", verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 학습 스텝(self.n_calls)이 render_freq 배수가 되면 이미지 저장
        if self.n_calls % self.render_freq == 0:
            env = self.model.get_env()  # VecEnv or Monitor
            # 만약 VecEnv라면, env.envs[0]처럼 접근해서 실제 gym env를 꺼낼 수도 있음
            # 여기서는 단일 환경이라는 가정하에 다음과 같이 가능:
            if hasattr(env, "envs"):
                raw_env = env.envs[0]
            else:
                raw_env = env

            # PyBullet에서 rgb_array 렌더
            img = raw_env.render(mode="rgb_array")
            if isinstance(img, np.ndarray):
                # img.shape: (height, width, 3)
                filename = os.path.join(self.save_dir, f"step_{self.n_calls}.png")
                # OpenCV: BGR 순서이므로 변환 (선택 사항)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, img_bgr)
                if self.verbose > 0:
                    print(f"[RenderCallback] Saved screenshot at step {self.n_calls}: {filename}")
        return True
