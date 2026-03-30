from typing import Optional

from core.constants import PlayerId


class Connection:
    def __init__(self, v1, v2, id=None):
        self.id = id
        self.v1 = v1
        self.v2 = v2
        self.owner: Optional[PlayerId] = None

    def is_occupied(self) -> bool:
        return self.owner is not None

    def connects(self, vertex) -> bool:
        return vertex == self.v1 or vertex == self.v2

    def other_vertex(self, vertex):
        if vertex == self.v1:
            return self.v2
        if vertex == self.v2:
            return self.v1
        raise ValueError("Vertex not part of this connection.")

    def can_build_road(self, player_id: PlayerId, settlement_positions: dict, road_positions: dict) -> bool:
        if self.is_occupied():
            return False
        # check connected: one end has settlement or road
        v1_has = self.v1.id in settlement_positions[player_id] or any(
            c.id in road_positions[player_id] for c in self.v1.edges
        )
        v2_has = self.v2.id in settlement_positions[player_id] or any(
            c.id in road_positions[player_id] for c in self.v2.edges
        )
        return v1_has or v2_has

    def build_road(self, player_id: PlayerId) -> None:
        if self.owner is not None:
            raise ValueError(f"Connection {self.id} already has a road.")
        self.owner = player_id

    def __repr__(self) -> str:
        owner = self.owner.name if self.owner is not None else None
        return f"Connection(id={self.id}, owner={owner})"