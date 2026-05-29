"""Compatibility helpers for older model assets loaded with newer NumPy."""

import builtins

import numpy as np


def ensure_numpy_legacy_aliases():
    """Restore removed NumPy scalar aliases expected by legacy pickles/chumpy."""
    legacy_aliases = {
        "bool": np.bool_,
        "int": getattr(np, "int_", builtins.int),
        "float": getattr(np, "float64", builtins.float),
        "complex": getattr(np, "complex128", builtins.complex),
        "object": getattr(np, "object_", builtins.object),
        "unicode": getattr(np, "str_", builtins.str),
        "str": getattr(np, "str_", builtins.str),
    }

    for alias, target in legacy_aliases.items():
        if alias not in np.__dict__:
            setattr(np, alias, target)
