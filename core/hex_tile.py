from core.constants import Resource


class HexTile:
    def __init__(self, resource: Resource = Resource.DESERT, number: int = None, id=None):
        self.resource = resource
        self.number = number
        self.id = id

        self.vertices = []
        self.has_robber = False

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