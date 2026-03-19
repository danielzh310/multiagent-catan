import random
from collections import deque

from core.agent_state import AgentState
from core.board_layout import BoardLayout
from core.constants import (
    ActionType,
    BuildingType,
    COST_BUILD_CITY,
    COST_BUILD_ROAD,
    COST_BUILD_SETTLEMENT,
    COST_BUY_DEV_CARD,
    DevCard,
    GameConfig,
    PlayerId,
    ROBBER_DICE_VALUE,
    Resource,
    VICTORY_POINTS_TARGET,
    cost_to_dict,
)
from core.constructions import Building
from core.helpers import (
    apply_cost,
    calculate_victory_points,
    can_afford,
    distribute_resources,
    has_reached_victory,
    move_robber,
    roll_dice,
)


SETTLEMENT_COST = cost_to_dict(COST_BUILD_SETTLEMENT)
ROAD_COST = cost_to_dict(COST_BUILD_ROAD)
CITY_COST = cost_to_dict(COST_BUILD_CITY)
DEV_CARD_COST = cost_to_dict(COST_BUY_DEV_CARD)


class CatanEngine:
    def __init__(self, seed=None, victory_target=VICTORY_POINTS_TARGET):
        if seed is not None:
            random.seed(seed)

        self.seed = seed
        self.victory_target = victory_target
        self.config = GameConfig(victory_points_target=victory_target)

        self.board = None
        self.players = {}
        self.player_order = []
        self.current_player_idx = 0

        self.turn_number = 0
        self.current_dice = None
        self.winner = None

        self.resource_bank = {}
        self.development_deck = deque()

        self.initial_placement_phase = True
        self.initial_settlements_placed = {}
        self.initial_roads_placed = {}

        self.robber_pending = False

        self.reset()

    def reset(self):
        self.board = BoardLayout(seed=self.seed)

        self.players = {
            PlayerId.WHITE: AgentState(PlayerId.WHITE),
            PlayerId.BLUE: AgentState(PlayerId.BLUE),
            PlayerId.ORANGE: AgentState(PlayerId.ORANGE),
            PlayerId.RED: AgentState(PlayerId.RED),
        }

        self.player_order = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]
        random.shuffle(self.player_order)

        self.current_player_idx = 0
        self.turn_number = 0
        self.current_dice = None
        self.winner = None

        self.resource_bank = {
            Resource.WOOD: 19,
            Resource.BRICK: 19,
            Resource.SHEEP: 19,
            Resource.WHEAT: 19,
            Resource.ORE: 19,
        }

        dev_cards = (
            [DevCard.KNIGHT] * 14
            + [DevCard.VICTORY_POINT] * 5
            + [DevCard.ROAD_BUILDING] * 2
            + [DevCard.YEAR_OF_PLENTY] * 2
            + [DevCard.MONOPOLY] * 2
        )
        random.shuffle(dev_cards)
        self.development_deck = deque(dev_cards)

        self.initial_placement_phase = True
        self.initial_settlements_placed = {pid: 0 for pid in self.players}
        self.initial_roads_placed = {pid: 0 for pid in self.players}

        self.robber_pending = False

        return self.get_state()

    def get_current_player_id(self):
        return self.player_order[self.current_player_idx]

    def get_current_player(self):
        return self.players[self.get_current_player_id()]

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.player_order)
        if self.current_player_idx == 0:
            self.turn_number += 1

    def get_state(self):
        return {
            "turn_number": self.turn_number,
            "current_player": self.get_current_player_id(),
            "dice": self.current_dice,
            "winner": self.winner,
            "initial_placement_phase": self.initial_placement_phase,
            "robber_pending": self.robber_pending,
        }

    def get_player_summary(self, player_id):
        player = self.players[player_id]
        return {
            "player_id": player_id,
            "resources": dict(player.resources),
            "victory_points": calculate_victory_points(player),
            "roads": list(player.roads),
            "dev_cards": list(player.dev_cards),
        }

    def check_winner(self):
        for player_id, player in self.players.items():
            player.victory_points = calculate_victory_points(player)
            if has_reached_victory(player, self.victory_target):
                self.winner = player_id
                return player_id
        return None

    def start_turn(self):
        if self.initial_placement_phase:
            return

        self.current_dice = roll_dice()

        if self.current_dice == ROBBER_DICE_VALUE:
            self.robber_pending = True
            return

        distribute_resources(self.board, self.players, self.current_dice)

    def end_turn(self):
        self.current_dice = None
        self.get_current_player().reset_turn_flags()
        self.next_player()

    def get_legal_actions(self, player_id=None):
        if player_id is None:
            player_id = self.get_current_player_id()

        player = self.players[player_id]
        actions = []

        if self.winner is not None:
            return actions

        if self.robber_pending:
            for tile in self.board.tiles:
                if not tile.has_robber:
                    actions.append({
                        "type": ActionType.MOVE_ROBBER,
                        "tile_id": tile.id,
                    })
            return actions

        if self.initial_placement_phase:
            for vertex in self.board.vertices:
                if vertex.can_place_settlement(player_id):
                    actions.append({
                        "type": ActionType.BUILD_SETTLEMENT,
                        "vertex_id": vertex.id,
                        "initial": True,
                    })

            for connection in self.board.connections:
                if connection.can_build_road(player_id):
                    actions.append({
                        "type": ActionType.BUILD_ROAD,
                        "connection_id": connection.id,
                        "initial": True,
                    })

            return actions

        if can_afford(player, SETTLEMENT_COST):
            for vertex in self.board.vertices:
                if vertex.can_place_settlement(player_id):
                    actions.append({
                        "type": ActionType.BUILD_SETTLEMENT,
                        "vertex_id": vertex.id,
                    })

        if can_afford(player, ROAD_COST):
            for connection in self.board.connections:
                if connection.can_build_road(player_id):
                    actions.append({
                        "type": ActionType.BUILD_ROAD,
                        "connection_id": connection.id,
                    })

        if can_afford(player, CITY_COST):
            for vertex in self.board.vertices:
                if (
                    vertex.building is not None
                    and vertex.building.owner == player_id
                    and vertex.building.type == BuildingType.SETTLEMENT
                ):
                    actions.append({
                        "type": ActionType.BUILD_CITY,
                        "vertex_id": vertex.id,
                    })

        if can_afford(player, DEV_CARD_COST) and len(self.development_deck) > 0:
            actions.append({
                "type": ActionType.BUY_DEV_CARD,
            })

        actions.append({
            "type": ActionType.END_TURN,
        })

        return actions

    def apply_action(self, action, player_id=None):
        if player_id is None:
            player_id = self.get_current_player_id()

        if player_id != self.get_current_player_id():
            raise ValueError("Attempted to play out of turn.")

        action_type = action["type"]
        info = {"action": action_type}
        reward = 0.0
        done = False

        if action_type == ActionType.BUILD_SETTLEMENT:
            self._apply_build_settlement(player_id, action)
            reward = 0.2

        elif action_type == ActionType.BUILD_ROAD:
            self._apply_build_road(player_id, action)
            reward = 0.1

        elif action_type == ActionType.BUILD_CITY:
            self._apply_build_city(player_id, action)
            reward = 0.35

        elif action_type == ActionType.BUY_DEV_CARD:
            self._apply_buy_dev_card(player_id)
            reward = 0.1

        elif action_type == ActionType.MOVE_ROBBER:
            self._apply_move_robber(action)
            reward = 0.05

        elif action_type == ActionType.END_TURN:
            self.end_turn()

        else:
            raise NotImplementedError(f"Unsupported action type: {action_type}")

        winner = self.check_winner()
        if winner is not None:
            done = True
            if winner == player_id:
                reward += 1.0

        return reward, done, info

    def _apply_build_settlement(self, player_id, action):
        vertex = self.board.get_vertex_by_id(action["vertex_id"])
        player = self.players[player_id]

        if not vertex.can_place_settlement(player_id):
            raise ValueError("Illegal settlement placement.")

        if not action.get("initial", False):
            if not can_afford(player, SETTLEMENT_COST):
                raise ValueError("Player cannot afford settlement.")
            apply_cost(player, SETTLEMENT_COST)

        building = Building(BuildingType.SETTLEMENT, player_id, vertex)
        vertex.place_building(building)
        player.buildings.append(building)

        if self.initial_placement_phase:
            self.initial_settlements_placed[player_id] += 1

    def _apply_build_road(self, player_id, action):
        connection = self.board.get_connection_by_id(action["connection_id"])
        player = self.players[player_id]

        if not connection.can_build_road(player_id):
            raise ValueError("Illegal road placement.")

        if not action.get("initial", False):
            if not can_afford(player, ROAD_COST):
                raise ValueError("Player cannot afford road.")
            apply_cost(player, ROAD_COST)

        connection.build_road(player_id)
        player.roads.append(connection.id)

        if self.initial_placement_phase:
            self.initial_roads_placed[player_id] += 1
            self._check_initial_phase_complete()

    def _apply_build_city(self, player_id, action):
        vertex = self.board.get_vertex_by_id(action["vertex_id"])
        player = self.players[player_id]

        if vertex.building is None:
            raise ValueError("No settlement to upgrade.")

        if vertex.building.owner != player_id:
            raise ValueError("Cannot upgrade another player's settlement.")

        if vertex.building.type != BuildingType.SETTLEMENT:
            raise ValueError("Only settlements can be upgraded to cities.")

        if not can_afford(player, CITY_COST):
            raise ValueError("Player cannot afford city.")

        apply_cost(player, CITY_COST)
        vertex.upgrade_to_city()

    def _apply_buy_dev_card(self, player_id):
        player = self.players[player_id]

        if len(self.development_deck) == 0:
            raise ValueError("No development cards left.")

        if not can_afford(player, DEV_CARD_COST):
            raise ValueError("Player cannot afford dev card.")

        apply_cost(player, DEV_CARD_COST)
        card = self.development_deck.popleft()
        player.add_dev_card(card)

        if card == DevCard.VICTORY_POINT:
            player.dev_victory_points += 1

    def _apply_move_robber(self, action):
        target_tile_id = action["tile_id"]
        move_robber(self.board, target_tile_id)
        self.robber_pending = False

    def _check_initial_phase_complete(self):
        complete = True

        for player_id in self.players:
            if self.initial_settlements_placed[player_id] < 1:
                complete = False
            if self.initial_roads_placed[player_id] < 1:
                complete = False

        if complete:
            self.initial_placement_phase = False

    def run_full_game(self, agent_map, max_turns=500):
        self.reset()

        while self.winner is None and self.turn_number < max_turns:
            current_player_id = self.get_current_player_id()

            if not self.initial_placement_phase and self.current_dice is None:
                self.start_turn()

            legal_actions = self.get_legal_actions(current_player_id)

            if len(legal_actions) == 0:
                self.end_turn()
                continue

            agent = agent_map[current_player_id]
            action = agent.select_action(self, legal_actions)

            self.apply_action(action, current_player_id)

        return self.winner, {
            "turns": self.turn_number,
            "winner": self.winner,
        }