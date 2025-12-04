#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from .earth_intersection_minimizer import EarthIntersectionMinimizer

_REGISTRY = {}


def register(name: str, eim: EarthIntersectionMinimizer) -> None:
    """
    Register an EarthIntersectionMinimizer by name.

    :param name: name of the minimizer
    :param eim: the minimizer
    """
    if name in _REGISTRY:
        raise ValueError(f"'{name}' already registered.")
    _REGISTRY[name] = eim


def get(name: str) -> EarthIntersectionMinimizer:
    """
    Return a registered minimizer by name, else raise KeyError.

    :return: the minimizer
    """
    return _REGISTRY[name]
