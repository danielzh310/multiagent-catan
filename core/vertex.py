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

    def can_place_settlement(self, player_id: PlayerId, settlement_positions: dict, road_positions: dict, require_road: bool = True) -> bool:
        if self.is_occupied():
            return False

        # check distance: no adjacent vertex has settlement
        for neighbor in self.neighbors:
            if any(neighbor.id in pos for pos in settlement_positions.values()):
                return False

        if require_road:
            # check connected by player's own road
            connected = any(
                conn.id in road_positions[player_id] for conn in self.edges
            )
            if not connected:
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