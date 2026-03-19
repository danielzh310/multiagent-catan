import random
from collections import defaultdict

from core.constants import BuildingType, COLLECTABLE_RESOURCES, Resource


def roll_dice():
    return random.randint(1, 6) + random.randint(1, 6)


def distribute_resources(board, players, dice_value):
    for tile in board.tiles:
        if not tile.produces_on_roll(dice_value):
            continue

        resource = tile.resource

        for vertex in tile.vertices:
            if vertex.building is None:
                continue

            player_id = vertex.building.owner
            player = players[player_id]

            if vertex.building.type == BuildingType.SETTLEMENT:
                player.add_resource(resource, 1)
            elif vertex.building.type == BuildingType.CITY:
                player.add_resource(resource, 2)


def calculate_victory_points(player):
    return player.update_victory_points()


def has_reached_victory(player, target=10):
    return calculate_victory_points(player) >= target


def can_afford(player, cost_dict):
    for resource, amount in cost_dict.items():
        if player.resources.get(resource, 0) < amount:
            return False
    return True


def apply_cost(player, cost_dict):
    for resource, amount in cost_dict.items():
        player.resources[resource] = player.resources.get(resource, 0) - amount


def add_resources(player, resource_dict):
    for resource, amount in resource_dict.items():
        if resource == Resource.DESERT:
            continue
        player.resources[resource] = player.resources.get(resource, 0) + amount


def resource_dict():
    return defaultdict(int)


def get_all_players_with_buildings(board):
    players = set()

    for vertex in board.vertices:
        if vertex.building is not None:
            players.add(vertex.building.owner)

    return players


def move_robber(board, target_tile_id):
    for tile in board.tiles:
        tile.set_robber(False)

    board.tiles[target_tile_id].set_robber(True)


def empty_resource_map():
    return {resource: 0 for resource in COLLECTABLE_RESOURCES}