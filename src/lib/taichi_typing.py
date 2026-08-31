from typing import TYPE_CHECKING, Any

import taichi as ti

if TYPE_CHECKING:
    TaichiF32 = float
    TaichiTemplate = Any
    TaichiVector3F32 = Any
else:
    TaichiF32 = ti.f32
    TaichiTemplate = ti.template()
    TaichiVector3F32 = ti.types.vector(3, ti.f32)
