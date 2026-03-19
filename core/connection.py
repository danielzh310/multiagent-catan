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

    def can_build_road(self, player_id: PlayerId) -> bool:
        return not self.is_occupied()

    def build_road(self, player_id: PlayerId) -> None:
        if self.owner is not None:
            raise ValueError(f"Connection {self.id} already has a road.")
        self.owner = player_id

    def __repr__(self) -> str:
        owner = self.owner.name if self.owner is not None else None
        return f"Connection(id={self.id}, owner={owner})"