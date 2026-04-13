from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Tuple

from core.constants import Resource

if TYPE_CHECKING:
    from core.connection import Connection
    from core.vertex import Vertex


class Port:
    def __init__(self, resource: Optional[Resource] = None, exchange_rate: int = 3, id: Optional[int] = None):
        self.resource: Optional[Resource] = resource
        self.exchange_rate: int = exchange_rate
        self.id: Optional[int] = id
        self.vertices: Tuple["Vertex", "Vertex"] | list["Vertex"] = []
        self.connection: Optional["Connection"] = None

    def is_generic(self) -> bool:
        return self.resource is None

    def is_specialized(self) -> bool:
        return self.resource is not None

    def matches_resource(self, resource: Resource) -> bool:
        return self.resource == resource

    def __repr__(self) -> str:
        resource_name = "GENERIC" if self.resource is None else self.resource.name
        return f"Port(id={self.id}, resource={resource_name}, rate={self.exchange_rate})"
