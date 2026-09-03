"""Dense global solve 최적화 전후를 비교하기 위한 임시 벤치마크.

실행:
    PYTHONPATH=src .venv/bin/python benchmarks/solver_performance.py

동일한 인자로 최적화 전후에 각각 실행해서 출력값을 비교한다. 이 파일은
정확성을 검증하는 테스트가 아니며, 실행 중인 다른 프로그램의 영향을 줄이기
위해 각 측정의 최솟값과 중앙값을 함께 출력한다.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np
import taichi as ti

from cloth.global_system import DenseGlobalSystem
from cloth.grid import create_cloth_grid
from cloth.rest_state import create_triangle_rest_state
from cloth.solver import ClothSolver


def measure(
    action: Callable[[], None],
    *,
    repetitions: int,
    samples: int,
) -> tuple[float, float]:
    """action 한 번당 소요 시간의 최솟값과 중앙값을 초 단위로 반환한다."""
    seconds_per_action: list[float] = []

    for _ in range(samples):
        ti.sync()
        started_at = time.perf_counter()
        for _ in range(repetitions):
            action()
        ti.sync()

        elapsed = time.perf_counter() - started_at
        seconds_per_action.append(elapsed / repetitions)

    return min(seconds_per_action), statistics.median(seconds_per_action)


def milliseconds(seconds: float) -> float:
    return seconds * 1_000.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense global solve의 최적화 전후 실행 시간을 비교합니다."
    )
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--global-solves", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    arguments = parser.parse_args()

    if arguments.grid_size < 2:
        parser.error("--grid-size는 2 이상이어야 합니다.")
    for argument_name in (
        "warmup_steps",
        "physics_steps",
        "global_solves",
        "samples",
    ):
        if getattr(arguments, argument_name) < 1:
            parser.error(f"--{argument_name.replace('_', '-')}는 1 이상이어야 합니다.")

    return arguments


def main() -> None:
    arguments = parse_arguments()
    ti.init(arch=ti.cpu, offline_cache=False)

    initial_positions, triangle_indices, _ = create_cloth_grid(
        arguments.grid_size,
        position=(0.0, 2.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    rest_state = create_triangle_rest_state(initial_positions, triangle_indices)

    construction_started_at = time.perf_counter()
    solver = ClothSolver(
        initial_positions=initial_positions,
        rest_state=rest_state,
        time_step=1.0 / 60.0,
    )
    construction_seconds = time.perf_counter() - construction_started_at

    zero_gravity = (0.0, 0.0, 0.0)
    for _ in range(arguments.warmup_steps):
        solver.step(gravity=zero_gravity)

    def build_dense_global_system() -> None:
        DenseGlobalSystem(
            vertex_count=len(initial_positions),
            time_step=1.0 / 60.0,
            constraints=solver.projective_constraints,
        )

    global_system_minimum, global_system_median = measure(
        build_dense_global_system,
        repetitions=1,
        samples=arguments.samples,
    )

    physics_minimum, physics_median = measure(
        lambda: solver.step(gravity=zero_gravity),
        repetitions=arguments.physics_steps,
        samples=arguments.samples,
    )

    # Global solve만 측정할 수 있도록 predicted position과 local projection을
    # 한 번 준비한다. 이후 측정에는 SVD local step과 상태 갱신이 포함되지 않는다.
    solver.reset()
    solver._predict_positions(solver._time_step, zero_gravity)
    solver._initialize_solve_positions()
    solver._local_step()

    def solve_without_prefactorization() -> None:
        """최적화 전 DenseGlobalSystem.solve와 같은 경로로 매번 LHS를 분해한다."""
        predicted_numpy = solver.predicted_positions.to_numpy().astype(np.float64)
        rhs_numpy = predicted_numpy / solver._time_step**2

        for constraint in solver.projective_constraints:
            constraint.add_rhs(rhs_numpy)

        solution_numpy = np.linalg.solve(
            solver.global_system.lhs_matrix,
            rhs_numpy,
        )
        solver.solve_positions.from_numpy(solution_numpy.astype(np.float32))

    baseline_global_minimum, baseline_global_median = measure(
        solve_without_prefactorization,
        repetitions=arguments.global_solves,
        samples=arguments.samples,
    )
    prefactorized_global_minimum, prefactorized_global_median = measure(
        solver._global_step,
        repetitions=arguments.global_solves,
        samples=arguments.samples,
    )

    triangle_count = len(triangle_indices) // 3
    print("\nDense global solve benchmark")
    print(f"  grid                 : {arguments.grid_size} x {arguments.grid_size}")
    print(f"  vertices             : {len(initial_positions)}")
    print(f"  triangles            : {triangle_count}")
    print(f"  solver iterations    : {solver._solver_iterations}")
    print(f"  samples              : {arguments.samples}")
    print(f"  solver construction  : {milliseconds(construction_seconds):.3f} ms")
    print(
        "  global system build  : "
        f"min {milliseconds(global_system_minimum):.3f} ms, "
        f"median {milliseconds(global_system_median):.3f} ms"
    )
    print(
        "  complete physics step: "
        f"min {milliseconds(physics_minimum):.3f} ms, "
        f"median {milliseconds(physics_median):.3f} ms"
    )
    print(
        "  global solve (before): "
        f"min {milliseconds(baseline_global_minimum):.3f} ms, "
        f"median {milliseconds(baseline_global_median):.3f} ms"
    )
    print(
        "  global solve (after) : "
        f"min {milliseconds(prefactorized_global_minimum):.3f} ms, "
        f"median {milliseconds(prefactorized_global_median):.3f} ms"
    )


if __name__ == "__main__":
    main()
