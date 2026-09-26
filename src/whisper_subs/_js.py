"""JavaScript semantics the jp-subs port has to reproduce exactly.

The prompts and segmentation are ported from jp-subs' JS and parity-tested
byte for byte (tests/test_parity.py), so these small differences matter.
"""

import math


def js_len(s: str) -> int:
    """String.prototype.length: UTF-16 code units, so 𠮷 counts as 2."""
    return len(s.encode("utf-16-le")) // 2


def js_round(x: float) -> int:
    """Math.round: halves go up (2.5 -> 3, -2.5 -> -2). Python's round() goes to
    even, and floor(x + 0.5) is off by one for 0.49999999999999994."""
    f = math.floor(x)
    return f + 1 if x - f >= 0.5 else f
