
import pytest
import subprocess
import os

@pytest.mark.integration
def test_train_ppo_script():
    # train_ppo.py 안에서 timesteps를 매우 작게 설정했거나,
    # 테스트용으로 인자를 던져서 --test 모드로 짧게 돌아가게 할 수도 있음
    script_path = os.path.join("scripts", "train_ppo.py")

    try:
        # 1) 실제 스크립트를 서브프로세스로 실행
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"train_ppo.py failed with error: {e}")
