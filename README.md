# RL_PYBULLET_SB3

이 저장소는 PyBullet 환경 + Stable Baselines3를 사용한 강화학습 예제 프로젝트입니다. 

PyBullet의 `HopperBulletEnv-v0` 환경을 예시로,  
Stable Baselines3의 PPO(A2C 계열) 알고리즘을 활용해 CPU에서 강화학습을 진행하는 프로젝트 구조 예시입니다.

주요 폴더 구조 및 파일:

- `configs/`: PPO 하이퍼파라미터 JSON 등 설정 파일
- `data/`: 학습 시 생성되는 로그나 중간데이터
- `experiments/`: 실험별 스크립트, 결과 정리
- `logs/`: TensorBoard 로그, SB3 모니터링 로그 등
- `models/`: 학습된 모델 및 Q-table 저장
- `scripts/`: 학습(`train_ppo.py`), 평가(`eval_ppo.py`) 등 실행 스크립트
- `tests/`: 단위 테스트 코드
- `utils/`: 재사용 유틸 함수
- `environment.yml`: conda 가상환경 정보
- `config.json`: 전체 프로젝트 공통 설정