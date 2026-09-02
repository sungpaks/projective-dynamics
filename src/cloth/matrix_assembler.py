from typing import Protocol

import numpy as np


class MatrixAssembler(Protocol):
    def add(
        self,
        row: int,
        column: int,
        value: float,
    ) -> None: ...


class DiagonalMatrixAssembler:
    """대각행렬 전용 MatrixAssembler. IdentityConstraintSet 검증용"""

    def __init__(self, size: int) -> None:
        self.values = np.zeros(size, dtype=np.float32)

    def add(
        self,
        row: int,
        column: int,
        value: float,
    ) -> None:
        if row != column:
            raise ValueError("DiagonalMatrixAssembler only supports diagonal entries.")

        self.values[row] += value


class DenseMatrixAssembler:
    def __init__(self, size: int) -> None:
        self.values = np.zeros(
            (size, size),
            dtype=np.float64,
        )

    def add(
        self,
        row: int,
        column: int,
        value: float,
    ) -> None:
        self.values[row, column] += value


# TODO: SparseMatrixAssembler
# TODO: add를 갖는 Protocol? 인터페이스?
