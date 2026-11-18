# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""torch.profiler ranges for the MiniMax H3 request path.

The helpers return context managers that only push and pop a RecordFunction
user scope (``torch.profiler.record_function``). They never touch tensors,
RNG state, streams or device synchronization, so wrapping code in them cannot
change numerics. Names are stable ``h3.*`` strings; per-step ranges end in
the step index so Perfetto can group them by prefix.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext

import torch
from torch.profiler import record_function

H3_PROFILE_DIT_DETAIL_ENV = "VLLM_OMNI_H3_PROFILE_DIT_DETAIL"


def _dit_detail_level() -> int:
    try:
        return int(os.environ.get(H3_PROFILE_DIT_DETAIL_ENV, "0") or "0")
    except ValueError:
        return 0


# Ranges inside MiniMaxH3DiTModel.forward are opt-in. Kineto attributes each
# GPU kernel to the innermost user range only, so any child range splits the
# ``h3.nfe.{i}`` span on the GPU track.
#   0 (default): no DiT-internal ranges; ``h3.nfe.{i}`` is one GPU span per NFE.
#   1: DiT sections (rope table, embed, SP prepare, AdaLN cache, blocks,
#      final layer + SP gather, output select).
#   2: level 1 plus one range per DiT block.
H3_PROFILE_DIT_DETAIL = _dit_detail_level()


def h3_range(name: str) -> AbstractContextManager:
    """A ``record_function`` range while a profiler is active; otherwise (or under torch.compile) a no-op.

    The fast path skips the RecordFunction dispatcher ops entirely, so the ranges cost nothing in
    production. No range straddles start/stop_profile: those RPCs run between requests in the worker
    busy loop.
    """
    if torch.compiler.is_compiling() or not torch.autograd._profiler_enabled():
        return nullcontext()
    return record_function(name)


def h3_dit_range(name: str, level: int = 1) -> AbstractContextManager:
    """An opt-in range inside the DiT forward (see ``H3_PROFILE_DIT_DETAIL``)."""
    if H3_PROFILE_DIT_DETAIL < level:
        return nullcontext()
    return h3_range(name)


__all__ = ["H3_PROFILE_DIT_DETAIL", "H3_PROFILE_DIT_DETAIL_ENV", "h3_dit_range", "h3_range"]
