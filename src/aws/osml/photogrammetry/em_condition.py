#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from abc import ABC, abstractmethod

from .coordinates import GeodeticWorldCoordinate


class ElevationModelCondition(ABC):
    """
    A base class functor for determining a True / False condition based on a given coordinate.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def is_true(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Return if the condition for supplied coordinate is True.

        :param world_coordinate: the coordinate to evaluate

        :return: True if condition passes, else False
        """


class EMConditionFalse(ElevationModelCondition):
    """
    An always False ElevationModel condition.
    """

    def __init__(self) -> None:
        pass

    def is_true(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Always returns False.

        :param world_coordinate: the coordinate to evaluate

        :return: False
        """
        return False


class EMConditionTrue(ElevationModelCondition):
    """
    An always True ElevationModel condition.
    """

    def __init__(self) -> None:
        pass

    def is_true(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Always returns True.

        :param world_coordinate: the coordinate to evaluate

        :return: True
        """
        return True
