from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple


class PlayerId(IntEnum):
    WHITE = 0
    BLUE = 1
    ORANGE = 2
    RED = 3


class Resource(IntEnum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4
    DESERT = 5


COLLECTABLE_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


class BuildingType(IntEnum):
    SETTLEMENT = 0
    CITY = 1


class DevCard(IntEnum):
    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    YEAR_OF_PLENTY = 3
    MONOPOLY = 4


class Phase(IntEnum):
    SETUP = 0
    TURN = 1
    GAME_OVER = 2


class ActionType(IntEnum):
    END_TURN = 0
    BUILD_ROAD = 1
    BUILD_SETTLEMENT = 2
    BUILD_CITY = 3
    BUY_DEV_CARD = 4
    PLAY_DEV_CARD = 5
    MOVE_ROBBER = 6
    TRADE_BANK = 7
    TRADE_PLAYER = 8
    DISCARD_CARDS = 9


ResourceCost = Tuple[int, int, int, int, int]

COST_BUILD_ROAD: ResourceCost = (1, 1, 0, 0, 0)
COST_BUILD_SETTLEMENT: ResourceCost = (1, 1, 1, 1, 0)
COST_BUILD_CITY: ResourceCost = (0, 0, 0, 2, 3)
COST_BUY_DEV_CARD: ResourceCost = (0, 0, 1, 1, 1)


VICTORY_POINTS_TARGET = 10

VP_SETTLEMENT = 1
VP_CITY = 2


NUM_TILES = 19
NUM_PORTS = 9
NUM_PLAYERS_DEFAULT = 4


TILE_RESOURCE_COUNTS: Dict[Resource, int] = {
    Resource.WOOD: 4,
    Resource.BRICK: 3,
    Resource.SHEEP: 4,
    Resource.WHEAT: 4,
    Resource.ORE: 3,
    Resource.DESERT: 1,
}


NUMBER_TOKEN_COUNTS: Dict[int, int] = {
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 1,
}


ROBBER_DICE_VALUE = 7


DEV_CARD_COUNTS: Dict[DevCard, int] = {
    DevCard.KNIGHT: 14,
    DevCard.VICTORY_POINT: 5,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
}


class PortType(IntEnum):
    GENERIC_3_TO_1 = 0
    WOOD_2_TO_1 = 1
    BRICK_2_TO_1 = 2
    SHEEP_2_TO_1 = 3
    WHEAT_2_TO_1 = 4
    ORE_2_TO_1 = 5


PORT_COUNTS: Dict[PortType, int] = {
    PortType.GENERIC_3_TO_1: 4,
    PortType.WOOD_2_TO_1: 1,
    PortType.BRICK_2_TO_1: 1,
    PortType.SHEEP_2_TO_1: 1,
    PortType.WHEAT_2_TO_1: 1,
    PortType.ORE_2_TO_1: 1,
}


def cost_to_dict(cost: ResourceCost) -> Dict[Resource, int]:
    return {
        Resource.WOOD: cost[0],
        Resource.BRICK: cost[1],
        Resource.SHEEP: cost[2],
        Resource.WHEAT: cost[3],
        Resource.ORE: cost[4],
    }


def empty_hand() -> Dict[Resource, int]:
    return {r: 0 for r in COLLECTABLE_RESOURCES}


@dataclass(frozen=True)
class GameConfig:
    num_players: int = NUM_PLAYERS_DEFAULT
    victory_points_target: int = VICTORY_POINTS_TARGET
    robber_dice_value: int = ROBBER_DICE_VALUE