from __future__ import annotations

from typing import List, Optional, Tuple

from core.constants import Resource


class HexTile:
    def __init__(
        self,
        resource: Resource = Resource.DESERT,
        number: Optional[int] = None,
        id: Optional[int] = None,
    ):
        self.resource: Resource = resource
        self.number: Optional[int] = number
        self.id: Optional[int] = id
        self.coord: Tuple[int, int] = (0, 0)
        self.vertices: List["Vertex"] = []
        self.has_robber: bool = False

    def is_desert(self) -> bool:
        return self.resource == Resource.DESERT

    def set_robber(self, value: bool = True) -> None:
        self.has_robber = value

    def produces_on_roll(self, dice_value: int) -> bool:
        if self.has_robber:
            return False
        if self.resource == Resource.DESERT:
            return False
        return self.number == dice_value

    def __repr__(self) -> str:
        return (
            f"HexTile(id={self.id}, "
            f"resource={self.resource.name}, "
            f"number={self.number}, "
            f"robber={self.has_robber})"
        )
