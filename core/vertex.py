from __future__ import annotations

from typing import List, Optional

from core.constructions import Building
from core.constants import BuildingType, PlayerId


class Vertex:
    def __init__(self, id=None):
        self.id = id
        self.edges: List = []
        self.tiles: List = []
        self.neighbors: List["Vertex"] = []
        self.building: Optional[Building] = None

    def is_occupied(self) -> bool:
        return self.building is not None

    def owner(self) -> Optional[PlayerId]:
        if self.building is None:
            return None
        return self.building.owner

    def has_player_building(self, player_id: PlayerId) -> bool:
        return self.building is not None and self.building.owner == player_id

    def has_opponent_building(self, player_id: PlayerId) -> bool:
        return self.building is not None and self.building.owner != player_id

    def can_place_settlement(
        self,
        player_id: PlayerId,
        settlement_positions: dict,
        road_positions: dict,
        city_positions: dict | None = None,
        require_road: bool = True,
    ) -> bool:
        """
        Rule-aligned settlement legality:
        - vertex must be empty
        - no adjacent vertex may contain any building
        - if not in setup, settlement must connect to one of the player's roads
        """
        if self.is_occupied():
            return False

        for neighbor in self.neighbors:
            if neighbor.is_occupied():
                return False

        if require_road:
            connected = any(conn.id in road_positions.get(player_id, set()) for conn in self.edges)
            if not connected:
                return False

        return True

    def can_upgrade_to_city(self, player_id: PlayerId) -> bool:
        if self.building is None:
            return False
        if self.building.owner != player_id:
            return False
        return self.building.type == BuildingType.SETTLEMENT

    def place_building(self, building: Building) -> None:
        if self.building is not None:
            raise ValueError(f"Vertex {self.id} already occupied.")
        self.building = building
        building.vertex = self

    def upgrade_to_city(self) -> None:
        if self.building is None:
            raise ValueError("No building to upgrade.")
        if self.building.type != BuildingType.SETTLEMENT:
            raise ValueError("Only a settlement can be upgraded to a city.")
        self.building.upgrade_to_city()

    def adjacent_player_roads(self, player_id: PlayerId, road_positions: dict) -> list[int]:
        return [conn.id for conn in self.edges if conn.id in road_positions.get(player_id, set())]

    def adjacent_opponent_building_blocks(self, player_id: PlayerId) -> bool:
        """
        Useful later for route-evaluation helpers.
        """
        return self.building is not None and self.building.owner != player_id

    def __repr__(self) -> str:
        owner = self.owner().name if self.owner() is not None else None
        building_type = self.building.type.name if self.building is not None else None
        return f"Vertex(id={self.id}, owner={owner}, building={building_type})"