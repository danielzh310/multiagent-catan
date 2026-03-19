from typing import List, Optional

from core.constructions import Building
from core.constants import PlayerId


class Vertex:
    def __init__(self, id=None):
        self.id = id
        self.edges: List = []
        self.tiles: List = []
        self.neighbors: List["Vertex"] = []
        self.building: Optional[Building] = None

    def is_occupied(self) -> bool:
        return self.building is not None

    def can_place_settlement(self, player_id: PlayerId) -> bool:
        if self.is_occupied():
            return False

        for neighbor in self.neighbors:
            if neighbor.building is not None:
                return False

        return True

    def place_building(self, building: Building) -> None:
        if self.building is not None:
            raise ValueError(f"Vertex {self.id} already occupied.")
        self.building = building
        building.vertex = self

    def upgrade_to_city(self) -> None:
        if self.building is None:
            raise ValueError("No building to upgrade.")
        self.building.upgrade_to_city()

    def owner(self) -> Optional[PlayerId]:
        if self.building is None:
            return None
        return self.building.owner

    def __repr__(self) -> str:
        owner = self.owner().name if self.owner() is not None else None
        return f"Vertex(id={self.id}, owner={owner})"