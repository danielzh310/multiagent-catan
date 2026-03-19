import copy

from core.constants import ActionType
from core.engine import CatanEngine


class CatanEnv:
    def __init__(self, seed=None, victory_target=10):
        self.seed = seed
        self.victory_target = victory_target
        self.engine = CatanEngine(seed=seed, victory_target=victory_target)

    def reset(self):
        self.engine.reset()
        return self.get_observation()

    def step(self, action):
        current_player = self.engine.get_current_player_id()
        reward, done, info = self.engine.apply_action(action, current_player)
        observation = self.get_observation()
        return observation, reward, done, info

    def get_current_player_id(self):
        return self.engine.get_current_player_id()

    def get_legal_actions(self, player_id=None):
        return self.engine.get_legal_actions(player_id)

    def get_observation(self, player_id=None):
        if player_id is None:
            player_id = self.engine.get_current_player_id()

        player = self.engine.players[player_id]

        observation = {
            "game": {
                "turn_number": self.engine.turn_number,
                "current_player": self.engine.get_current_player_id(),
                "dice": self.engine.current_dice,
                "winner": self.engine.winner,
                "initial_placement_phase": self.engine.initial_placement_phase,
                "robber_pending": self.engine.robber_pending,
            },
            "player": {
                "player_id": player_id,
                "resources": dict(player.resources),
                "victory_points": player.victory_points,
                "roads": list(player.roads),
                "dev_cards": list(player.dev_cards),
                "dev_victory_points": player.dev_victory_points,
            },
            "players": self._build_players_view(),
            "board": {
                "tiles": self._build_tile_view(),
                "vertices": self._build_vertex_view(),
                "connections": self._build_connection_view(),
                "ports": self._build_port_view(),
            },
            "legal_actions": self.get_legal_actions(player_id),
        }

        return observation

    def _build_players_view(self):
        players_view = {}

        for player_id, player in self.engine.players.items():
            players_view[player_id] = {
                "player_id": player_id,
                "resources": dict(player.resources),
                "victory_points": player.victory_points,
                "roads": list(player.roads),
                "dev_card_count": len(player.dev_cards),
                "dev_victory_points": player.dev_victory_points,
                "building_count": len(player.buildings),
            }

        return players_view

    def _build_tile_view(self):
        tiles = []

        for tile in self.engine.board.tiles:
            tiles.append({
                "id": tile.id,
                "resource": tile.resource,
                "number": tile.number,
                "has_robber": tile.has_robber,
                "vertex_ids": [vertex.id for vertex in tile.vertices],
            })

        return tiles

    def _build_vertex_view(self):
        vertices = []

        for vertex in self.engine.board.vertices:
            if vertex.building is None:
                building_type = None
                owner = None
            else:
                building_type = vertex.building.type
                owner = vertex.building.owner

            vertices.append({
                "id": vertex.id,
                "building_type": building_type,
                "owner": owner,
                "neighbor_ids": [neighbor.id for neighbor in vertex.neighbors],
                "tile_ids": [tile.id for tile in vertex.tiles],
                "edge_ids": [edge.id for edge in vertex.edges],
            })

        return vertices

    def _build_connection_view(self):
        connections = []

        for connection in self.engine.board.connections:
            connections.append({
                "id": connection.id,
                "v1": connection.v1.id,
                "v2": connection.v2.id,
                "owner": connection.owner,
            })

        return connections

    def _build_port_view(self):
        ports = []

        for port in self.engine.board.ports:
            ports.append({
                "id": port.id,
                "resource": port.resource,
                "exchange_rate": port.exchange_rate,
            })

        return ports

    def get_action_mask(self, player_id=None):
        if player_id is None:
            player_id = self.engine.get_current_player_id()

        legal_actions = self.engine.get_legal_actions(player_id)

        mask = {
            ActionType.BUILD_SETTLEMENT: 0,
            ActionType.BUILD_ROAD: 0,
            ActionType.BUILD_CITY: 0,
            ActionType.BUY_DEV_CARD: 0,
            ActionType.MOVE_ROBBER: 0,
            ActionType.END_TURN: 0,
        }

        for action in legal_actions:
            mask[action["type"]] = 1

        return mask

    def clone_state(self):
        return copy.deepcopy(self.engine)

    def restore_state(self, engine_state):
        self.engine = copy.deepcopy(engine_state)

    def render(self):
        return self.get_observation()

    def sample_random_action(self, player_id=None):
        legal_actions = self.get_legal_actions(player_id)
        if not legal_actions:
            return None
        return legal_actions[0]