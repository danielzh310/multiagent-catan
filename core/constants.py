# core/constants.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, Tuple


class Resource(IntEnum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4
    DESERT = 5 # Note: Desert isn't a collectable resource but is useful for tile typing

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
    # Phases of the game are general for now as a placeholder
    SETUP = 0
    TURN = 1
    GAME_OVER = 2


class ActionType(IntEnum):
    # Not all the possible actions, just some atm
    END_TURN = 0
    BUILD_ROAD = 1
    BUILD_SETTLEMENT = 2
    UPGRADE_CITY = 3
    BUY_DEV_CARD = 4
    PLAY_DEV_CARD = 5
    MOVE_ROBBER = 6
    TRADE_BANK = 7
    TRADE_PLAYER = 8

# Costs are expressed as vectors over COLLECTABLE_RESOURCES order
# (wood, brick, sheep, wheat, ore)
ResourceCost = Tuple[int, int, int, int, int]

COST_BUILD_ROAD: ResourceCost = (1, 1, 0, 0, 0)
COST_BUILD_SETTLEMENT: ResourceCost = (1, 1, 1, 1, 0)
COST_UPGRADE_CITY: ResourceCost = (0, 0, 0, 2, 3)
COST_BUY_DEV_CARD: ResourceCost = (0, 0, 1, 1, 1)

VICTORY_POINTS_TARGET = 10

VP_SETTLEMENT = 1
VP_CITY = 2

# Building the board composition for standard Catan

NUM_TILES = 19
NUM_PORTS = 9
NUM_PLAYERS_DEFAULT = 4

# Standard resource tile counts, this excludes desert
TILE_RESOURCE_COUNTS: Dict[Resource, int] = {
    Resource.WOOD: 4,
    Resource.BRICK: 3,
    Resource.SHEEP: 4,
    Resource.WHEAT: 4,
    Resource.ORE: 3,
    Resource.DESERT: 1,
}

# Standard number token counts for 19-tile board should be 18 tokens total
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

# Standard dev card deck composition
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
    """Convert wood, brick, sheep, wheat, ore vector to a resource to count dict."""
    return {
        Resource.WOOD: cost[0],
        Resource.BRICK: cost[1],
        Resource.SHEEP: cost[2],
        Resource.WHEAT: cost[3],
        Resource.ORE: cost[4],
    }

def empty_hand() -> Dict[Resource, int]:
    """Create a zero count resource hand dict for collectable resources."""
    return {r: 0 for r in COLLECTABLE_RESOURCES}


@dataclass(frozen=True)
class GameConfig:
    """Central place for rules, expand for later."""
    num_players: int = NUM_PLAYERS_DEFAULT
    victory_points_target: int = VICTORY_POINTS_TARGET
    robber_dice_value: int = ROBBER_DICE_VALUE