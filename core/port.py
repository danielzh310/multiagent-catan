from core.constants import Resource


class Port:
    def __init__(self, resource: Resource = None, exchange_rate: int = 3, id=None):
        self.resource = resource
        self.exchange_rate = exchange_rate
        self.id = id
        self.vertices = []
        self.connection = None

    def is_generic(self) -> bool:
        return self.resource is None

    def is_specialized(self) -> bool:
        return self.resource is not None

    def matches_resource(self, resource: Resource) -> bool:
        return self.resource == resource

    def __repr__(self) -> str:
        resource_name = "GENERIC" if self.resource is None else self.resource.name
        return f"Port(id={self.id}, resource={resource_name}, rate={self.exchange_rate})"