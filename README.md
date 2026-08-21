# Projective Dynamics with Taichi

Python과 Taichi로 Projective Dynamics를 단계별로 구현하는 학습 프로젝트입니다.

## 실행 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 첫 번째 예제: 정적 천 메쉬

```bash
python src/cloth_viewer.py
```

마우스 오른쪽 버튼을 누른 채 움직이면 카메라를 회전할 수 있고, `W`, `A`, `S`, `D`, `E`, `Q`로 이동할 수 있습니다. 창을 닫으면 프로그램이 종료됩니다.

현재 예제는 물리를 계산하지 않습니다. NumPy로 격자 메쉬를 만든 뒤 Taichi 필드에 복사하고 GGUI로 렌더링하는 첫 단계입니다.

