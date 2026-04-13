from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from core.constants import PlayerId

if TYPE_CHECKING:
    from core.vertex import Vertex


class Connection:
    def __init__(self, v1: "Vertex", v2: "Vertex", id: Optional[int] = None):
        self.id: Optional[int] = id
        self.v1: Vertex = v1
        self.v2: Vertex = v2
        self.owner: Optional[PlayerId] = None

    def is_occupied(self) -> bool:
        return self.owner is not None

    def connects(self, vertex: "Vertex") -> bool:
        return vertex == self.v1 or vertex == self.v2

    def other_vertex(self, vertex: "Vertex") -> "Vertex":
        if vertex == self.v1:
            return self.v2
        if vertex == self.v2:
            return self.v1
        raise ValueError("Vertex not part of this connection.")

    def _vertex_has_player_building(self, vertex: "Vertex", player_id: PlayerId, settlement_positions: dict, city_positions: dict | None = None) -> bool:
        if vertex.id in settlement_positions.get(player_id, set()):
            return True
        if city_positions is not None and vertex.id in city_positions.get(player_id, set()):
            return True
        return False

    def _vertex_has_opponent_building(self, vertex: "Vertex", player_id: PlayerId, settlement_positions: dict, city_positions: dict | None = None) -> bool:
        for pid, positions in settlement_positions.items():
            if pid != player_id and vertex.id in positions:
                return True

        if city_positions is not None:
            for pid, positions in city_positions.items():
                if pid != player_id and vertex.id in positions:
                    return True

        return False

    def _vertex_has_adjacent_player_road(self, vertex: "Vertex", player_id: PlayerId, road_positions: dict) -> bool:
        for conn in vertex.edges:
            if conn.id == self.id:
                continue
            if conn.id in road_positions.get(player_id, set()):
                return True
        return False

    def can_build_road(
        self,
        player_id: PlayerId,
        settlement_positions: dict,
        road_positions: dict,
        city_positions: dict | None = None,
    ) -> bool:
        """
        Catan road legality, simplified but rule-aligned:

        - edge must be empty
        - road must connect to one of the player's existing roads or buildings
        - you may not extend through an opponent building
        """
        if self.is_occupied():
            return False

        v1_has_building = self._vertex_has_player_building(self.v1, player_id, settlement_positions, city_positions)
        v2_has_building = self._vertex_has_player_building(self.v2, player_id, settlement_positions, city_positions)

        if v1_has_building or v2_has_building:
            return True

        v1_has_road = self._vertex_has_adjacent_player_road(self.v1, player_id, road_positions)
        v2_has_road = self._vertex_has_adjacent_player_road(self.v2, player_id, road_positions)

        v1_blocked = self._vertex_has_opponent_building(self.v1, player_id, settlement_positions, city_positions)
        v2_blocked = self._vertex_has_opponent_building(self.v2, player_id, settlement_positions, city_positions)

        can_extend_from_v1 = v1_has_road and not v1_blocked
        can_extend_from_v2 = v2_has_road and not v2_blocked

        return can_extend_from_v1 or can_extend_from_v2

    def build_road(self, player_id: PlayerId) -> None:
        if self.owner is not None:
            raise ValueError(f"Connection {self.id} already has a road.")
        self.owner = player_id

    def blocks_player_route_at_vertex(
        self,
        vertex: "Vertex",
        player_id: PlayerId,
        settlement_positions: dict,
        city_positions: dict | None = None,
    ) -> bool:
        """
        Helper for longest-route logic:
        an opponent building on a vertex blocks route continuity through that vertex.
        """
        return self._vertex_has_opponent_building(vertex, player_id, settlement_positions, city_positions)

    def __repr__(self) -> str:
        owner = self.owner.name if self.owner is not None else None
        return f"Connection(id={self.id}, owner={owner}, v1={self.v1.id}, v2={self.v2.id})"
