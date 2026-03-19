from core.constants import BuildingType, PlayerId


class Building:
    def __init__(self, building_type: BuildingType, owner: PlayerId, vertex=None):
        self.type = building_type
        self.owner = owner
        self.vertex = vertex

    def is_settlement(self) -> bool:
        return self.type == BuildingType.SETTLEMENT

    def is_city(self) -> bool:
        return self.type == BuildingType.CITY

    def upgrade_to_city(self) -> None:
        self.type = BuildingType.CITY

    def __repr__(self) -> str:
        vertex_id = getattr(self.vertex, "id", None)
        return f"Building(type={self.type.name}, owner={self.owner.name}, vertex={vertex_id})"