# Projective Dynamics with Taichi

Python과 Taichi로 [Projective Dynamics](https://users.cs.utah.edu/~ladislav/bouaziz14projective/bouaziz14projective.pdf)를 구현하는 학습 프로젝트입니다.

## 실행 준비

macOS에서 + python3.11 기준:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

(2026.09.04기준)`cloth_viewer`는 `ti.metal`을 사용하므로 macOS/Metal 환경에서 실행가능

## 첫 번째 예제: 2D Cloth

```bash
python src/cloth_viewer.py
```

- 카메라 회전: 마우스 오른쪽 버튼 드래그
- 카메라 이동: `W`, `A`, `S`, `D`, `E`, `Q`
- 종료: 창 닫기

### 구현된 기능 (UPDATE: 20260904)

- 평면 메쉬: 텍스처 없이 2D 평면 메쉬를 절차적으로 생성, 해상도 조정 가능 (`GRID_SIZE`)
- Projective Dynamics의 Local-Global solver
- Strain Constraint
- Positional Constraint (상단 정점 고정)
- 화면 공간에 마우스 클릭(hold)하여 외력(바람-like) 생성
- Taichi Rendering

#### 최적화

- [x] Dense Matrix + Cholesky prefactorization
- [ ] Dense Matrix를 Sparse Matrix로 개선
- [ ] CPU/GPU 연산 경계
- [ ] ...

#### 마일스톤 (TODO)

- [ ] 질량 추가하기
- [ ] Bending Constraint
- [ ] Collision Detection 및 Collision에 의한 Constraint
- [ ] ...

### 실행 후 조작법

- 실행 직후: gravity 등 외력 없이 rest state
- 패널에서 `gravity` 체크박스로 중력 활성화
- 마우스 왼쪽 클릭(HOLD): 카메라가 바라보는 방향으로 외력 생성(바람)
- 마우스 오른쪽 클릭(DRAG): 카메라 회전
- `W`, `A`, `S`, `D`, `E`, `Q`: 카메라 이동
- 패널에서 `reset`: 초기 위치 복원 및 중력 off
- 패널에서 slider 조정: 외력의 강도(바람의 세기) 및 반지름 조정

## 저장할 때 자동 재실행

```bash
watchfiles --filter python 'python src/cloth_viewer.py' src
```

저장하면 프로세스 전체가 다시 시작되므로 창과 카메라 상태도 초기화

## 코드 품질과 테스트

```bash
ruff check .
ruff format --check .
pytest
```
