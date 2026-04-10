from __future__ import annotations

import random
from typing import Dict, List, Optional

from core.agent_state import AgentState
from core.board_layout import BoardLayout
from core.constants import (
    BuildingType,
    COST_BUILD_CITY,
    COST_BUY_DEV_CARD,
    COST_BUILD_ROAD,
    COLLECTABLE_RESOURCES,
    COST_BUILD_SETTLEMENT,
    DEV_CARD_COUNTS,
    DevCard,
    LARGEST_ARMY_MIN_KNIGHTS,
    PlayerId,
    Resource,
    VP_LARGEST_ARMY,
    RESOURCE_SUPPLY_COUNTS,
    VP_LONGEST_ROUTE,
    VICTORY_POINTS_TARGET,
    LONGEST_ROUTE_MIN_LENGTH,
    cost_to_dict,
)
from core.constructions import Building
from core.helpers import move_robber, roll_dice
from core.phase_router import PhaseRouter, TurnPhase
from core.trade_history import TradeHistory
from core.trade_manager import TradeManager, TradeResponse


class CatanEngine:
    def __init__(self, seed: Optional[int] = None, enable_trading: bool = True):
        self.random = random.Random(seed)
        self.enable_trading = enable_trading

        self.players: Dict[PlayerId, AgentState] = {
            PlayerId.WHITE: AgentState(PlayerId.WHITE),
            PlayerId.BLUE: AgentState(PlayerId.BLUE),
            PlayerId.ORANGE: AgentState(PlayerId.ORANGE),
            PlayerId.RED: AgentState(PlayerId.RED),
        }

        self.player_order: List[PlayerId] = [
            PlayerId.WHITE,
            PlayerId.BLUE,
            PlayerId.ORANGE,
            PlayerId.RED,
        ]

        self.current_player_idx = 0
        self.turn_number = 0

        self.initial_placement_phase = False
        self.initial_placement_index = 0
        self.initial_placement_stage = "settlement"
        self.initial_placement_order: List[PlayerId] = [] # Will be populated in reset()

        self.robber_pending = False
        self.last_roll: Optional[int] = None
        self.last_robber_event: Optional[dict] = None

        self.trade_history = TradeHistory()
        self.trade_manager = TradeManager(self.trade_history)
        self.phase_router = PhaseRouter()

        self.board = BoardLayout(seed=seed)
        self.settlement_positions: Dict[PlayerId, set[int]] = {p: set() for p in self.players}
        self.city_positions: Dict[PlayerId, set[int]] = {p: set() for p in self.players}
        self.road_positions: Dict[PlayerId, set[int]] = {p: set() for p in self.players}
        self.dev_card_deck: List[DevCard] = []
        self.longest_road_owner: Optional[PlayerId] = None
        self.largest_army_owner: Optional[PlayerId] = None
        self.robber_discard_required: Dict[PlayerId, int] = {}
        self.robber_discard_queue: List[PlayerId] = []
        self.robber_move_pending_player: Optional[PlayerId] = None

        self.winner: Optional[PlayerId] = None
        self.resource_bank = dict(RESOURCE_SUPPLY_COUNTS)

    def reset(self):
        for player in self.players.values():
            player.reset_for_new_game()
            player.resources = {
                Resource.WOOD: 0,
                Resource.BRICK: 0,
                Resource.SHEEP: 0,
                Resource.WHEAT: 0,
                Resource.ORE: 0,
            }
            player.update_victory_points()

        self.current_player_idx = 0
        self.turn_number = 0
        
        # Shuffle the main player order for the entire game
        self.random.shuffle(self.player_order)
        self.initial_placement_order = list(self.player_order) + list(reversed(self.player_order))

        self.initial_placement_phase = True
        self.initial_placement_index = 0
        self.initial_placement_stage = "settlement"

        self.robber_pending = False
        self.last_roll = None
        self.last_robber_event = None

        self.trade_manager.reset()
        self.phase_router.reset()
        self.phase_router.set_phase(TurnPhase.SETUP)

        self.settlement_positions = {p: set() for p in self.players}
        self.city_positions = {p: set() for p in self.players}
        self.road_positions = {p: set() for p in self.players}
        self.dev_card_deck = self._build_dev_card_deck()
        self.longest_road_owner = None
        self.largest_army_owner = None
        self.robber_discard_required = {}
        self.robber_discard_queue = []
        self.robber_move_pending_player = None

        for vertex in self.board.vertices:
            vertex.building = None

        for connection in self.board.connections:
            connection.owner = None

        desert = self.board.get_desert_tile()
        if desert is not None:
            self.board.move_robber_to_tile(desert.id)

        self.winner = None
        self.resource_bank = dict(RESOURCE_SUPPLY_COUNTS)

    def get_valid_settlement_vertices(self, player_id: PlayerId, require_road: bool = True) -> List[int]:
        return self.board.get_valid_settlement_vertices(
            player_id,
            self.settlement_positions,
            self.road_positions,
            city_positions=self.city_positions,
            require_road=require_road,
        )

    def get_valid_road_connections(self, player_id: PlayerId) -> List[int]:
        return self.board.get_valid_road_connections(
            player_id,
            self.settlement_positions,
            self.road_positions,
            city_positions=self.city_positions,
        )

    def get_current_player_id(self) -> PlayerId:
        if self.initial_placement_phase:
            idx = min(self.initial_placement_index, len(self.initial_placement_order) - 1)
            return self.initial_placement_order[idx]
        if self.robber_discard_queue:
            return self.robber_discard_queue[0]
        if self.robber_move_pending_player is not None:
            return self.robber_move_pending_player
        return self.player_order[self.current_player_idx]

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.player_order)
        if self.current_player_idx == 0:
            self.turn_number += 1
        self.players[self.player_order[self.current_player_idx]].reset_turn_flags()

    def _resource_total(self, player_id: PlayerId) -> int:
        return sum(int(v) for v in self.players[player_id].resources.values())

    def _resource_diversity(self, player_id: PlayerId) -> int:
        return sum(1 for v in self.players[player_id].resources.values() if int(v) > 0)

    def _build_dev_card_deck(self) -> List[DevCard]:
        deck: List[DevCard] = []
        for card, count in DEV_CARD_COUNTS.items():
            deck.extend([card] * int(count))
        self.random.shuffle(deck)
        return deck

    def _can_afford_resources(self, resources: Dict[Resource, int], cost: tuple) -> bool:
        cost_dict = cost_to_dict(cost)
        for resource, amount in cost_dict.items():
            if resources.get(resource, 0) < amount:
                return False
        return True

    def _build_readiness_score(self, resources: Dict[Resource, int]) -> float:
        score = 0.0

        if self._can_afford_resources(resources, COST_BUILD_SETTLEMENT):
            score += 1.0
        if self._can_afford_resources(resources, COST_BUILD_CITY):
            score += 1.4
        if self._can_afford_resources(resources, COST_BUY_DEV_CARD):
            score += 0.8

        return score

    def _best_maritime_rate(self, player_id: PlayerId, resource: Resource) -> int:
        best_rate = 4
        owned_vertices = set(self.settlement_positions[player_id]) | set(self.city_positions[player_id])
        for vertex_id in owned_vertices:
            port = self.board.get_port_for_vertex(vertex_id)
            if port is None:
                continue
            if port.is_generic():
                best_rate = min(best_rate, int(port.exchange_rate))
            elif port.matches_resource(resource):
                best_rate = min(best_rate, int(port.exchange_rate))
        return best_rate

    def _can_take_high_value_action(self, player_id: PlayerId) -> bool:
        player = self.players[player_id]

        can_build_settlement = (
            player.n_settlements < 5
            and player.can_pay_cost(COST_BUILD_SETTLEMENT)
            and bool(self.get_valid_settlement_vertices(player_id))
        )
        can_build_city = (
            player.n_settlements > 0
            and player.n_cities < 4
            and player.can_pay_cost(COST_BUILD_CITY)
            and bool(self.settlement_positions[player_id])
        )
        can_buy_dev = player.can_pay_cost(COST_BUY_DEV_CARD) and bool(self.dev_card_deck)

        return can_build_settlement or can_build_city or can_buy_dev

    def _trade_value(self, trade_vector: Dict[Resource, int]) -> int:
        return sum(int(v) for v in trade_vector.values())

    def _vp(self, player_id: PlayerId) -> int:
        return int(self.players[player_id].update_victory_points())

    def _attempt_pay_cost(self, player: AgentState, cost: tuple) -> bool:
        if player.can_pay_cost(cost):
            cost_dict = cost_to_dict(cost)
            player.pay_cost(cost)
            for resource, amount in cost_dict.items():
                self.resource_bank[resource] += amount
            player.update_victory_points()
            return True
        return False

    def _draw_dev_card(self) -> Optional[DevCard]:
        if not self.dev_card_deck:
            return None
        return self.dev_card_deck.pop()

    def _choose_monopoly_resource(self, player_id: PlayerId) -> Resource:
        best_resource = Resource.WOOD
        best_total = -1
        for resource in (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ):
            total = 0
            for opponent_id, opponent in self.players.items():
                if opponent_id == player_id:
                    continue
                total += int(opponent.resources.get(resource, 0))
            if total > best_total:
                best_total = total
                best_resource = resource
        return best_resource

    def _choose_invention_resources(self, player_id: PlayerId) -> List[Resource]:
        player = self.players[player_id]
        deficits = []
        for resource in (Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE):
            amount = int(player.resources.get(resource, 0))
            target = 1
            if resource in (Resource.WHEAT, Resource.ORE):
                target = 2 if resource == Resource.WHEAT else 3
            deficits.append((max(target - amount, 0), resource))
        deficits.sort(key=lambda item: (-item[0], int(item[1])))
        chosen = [deficits[0][1], deficits[1][1]]
        return chosen

    def _choose_robber_target_tile(self, player_id: PlayerId) -> Optional[int]:
        current_tile = next((tile.id for tile in self.board.tiles if tile.has_robber), None)
        best_tile_id = None
        best_score = float("-inf")

        for tile in self.board.tiles:
            if tile.id == current_tile:
                continue

            score = 0.0
            for vertex in tile.vertices:
                building = vertex.building
                if building is None:
                    continue
                if building.owner == player_id:
                    score -= 2.0
                elif building.type == BuildingType.CITY:
                    score += 2.5
                else:
                    score += 1.5

            if score > best_score:
                best_score = score
                best_tile_id = tile.id

        return best_tile_id

    def _build_free_road(self, player_id: PlayerId, connection_id: Optional[int] = None) -> bool:
        player = self.players[player_id]
        if player.n_roads >= 15:
            return False

        valid_roads = self.get_valid_road_connections(player_id)
        if not valid_roads:
            return False

        if connection_id is not None:
            if connection_id not in valid_roads:
                return False
            conn_id = connection_id
        else:
            conn_id = valid_roads[0]
        player.n_roads += 1
        player.roads.append(conn_id)
        self.road_positions[player_id].add(conn_id)
        connection = self.board.get_connection_by_id(conn_id)
        connection.build_road(player_id)
        self._update_special_awards()
        return True

    def _play_dev_card(self, player_id: PlayerId, card: DevCard, action: Optional[dict] = None) -> float:
        player = self.players[player_id]
        if not player.can_play_dev_card(card):
            return -0.02

        player.play_dev_card(card)

        if card == DevCard.KNIGHT:
            player.played_knights += 1
            target_tile_id = action.get("tile") if action is not None else None
            victim_id = action.get("victim") if action is not None else None
            self._move_robber_for_knight(player_id, target_tile_id=target_tile_id, victim_id=victim_id)
            self._update_special_awards()
            return 0.10

        if card == DevCard.ROAD_BUILDING:
            built = 0
            requested = []
            if action is not None:
                requested = [action.get("connection_1"), action.get("connection_2")]
            for connection_id in requested:
                if connection_id is None:
                    continue
                if self._build_free_road(player_id, connection_id=int(connection_id)):
                    built += 1
            while built < 2 and self._build_free_road(player_id):
                built += 1
            self._update_special_awards()
            return 0.03 * built

        if card == DevCard.INVENTION:
            resources = []
            if action is not None:
                for key in ("resource_1", "resource_2"):
                    raw = action.get(key)
                    if raw is None:
                        continue
                    try:
                        resources.append(raw if isinstance(raw, Resource) else Resource(raw))
                    except (TypeError, ValueError):
                        pass
            if len(resources) != 2:
                resources = self._choose_invention_resources(player_id)
            for resource in resources:
                if self.resource_bank.get(resource, 0) > 0:
                    player.add_resource(resource, 1)
                    self.resource_bank[resource] -= 1
            return 0.08

        if card == DevCard.MONOPOLY:
            resource = None
            if action is not None:
                raw = action.get("resource")
                try:
                    resource = raw if isinstance(raw, Resource) else Resource(raw)
                except (TypeError, ValueError):
                    resource = None
            if resource is None:
                resource = self._choose_monopoly_resource(player_id)
            total_taken = 0
            for opponent_id, opponent in self.players.items():
                if opponent_id == player_id:
                    continue
                amount = int(opponent.resources.get(resource, 0))
                if amount <= 0:
                    continue
                opponent.resources[resource] -= amount
                player.resources[resource] += amount
                total_taken += amount
            return 0.03 + 0.01 * total_taken

        return -0.02

    def _advance_after_main_action(self):
        if self.enable_trading:
            self.phase_router.complete_main_action_phase(self)
        else:
            self.phase_router.set_phase(TurnPhase.END_TURN)

    def _longest_road_length(self, player_id: PlayerId) -> int:
        player_roads = set(self.road_positions[player_id])
        best = 0

        def dfs(vertex, used_edges: set[int]) -> int:
            max_length = 0
            blocked = False
            building = vertex.building
            if building is not None and building.owner != player_id:
                blocked = True

            for conn in vertex.edges:
                if conn.id not in player_roads or conn.id in used_edges:
                    continue
                next_vertex = conn.other_vertex(vertex)
                extension = 1
                if not blocked:
                    extension += dfs(next_vertex, used_edges | {conn.id})
                max_length = max(max_length, extension)
            return max_length

        for conn_id in player_roads:
            conn = self.board.get_connection_by_id(conn_id)
            best = max(best, dfs(conn.v1, {conn_id}))
            best = max(best, dfs(conn.v2, {conn_id}))

        return best

    def _update_special_awards(self) -> None:
        for player in self.players.values():
            player.bonus_vp = 0

        road_lengths = {player_id: self._longest_road_length(player_id) for player_id in self.players}
        for player_id, length in road_lengths.items():
            self.players[player_id].longest_road_length = length

        eligible_roads = {
            player_id: length for player_id, length in road_lengths.items()
            if length >= LONGEST_ROUTE_MIN_LENGTH
        }
        longest_owner = None
        if eligible_roads:
            best_length = max(eligible_roads.values())
            leaders = [player_id for player_id, length in eligible_roads.items() if length == best_length]
            if len(leaders) == 1:
                longest_owner = leaders[0]
            elif self.longest_road_owner in leaders:
                longest_owner = self.longest_road_owner

        army_sizes = {
            player_id: player.played_knights
            for player_id, player in self.players.items()
            if player.played_knights >= LARGEST_ARMY_MIN_KNIGHTS
        }
        largest_army_owner = None
        if army_sizes:
            best_army = max(army_sizes.values())
            leaders = [player_id for player_id, size in army_sizes.items() if size == best_army]
            if len(leaders) == 1:
                largest_army_owner = leaders[0]
            elif self.largest_army_owner in leaders:
                largest_army_owner = self.largest_army_owner

        self.longest_road_owner = longest_owner
        self.largest_army_owner = largest_army_owner

        if longest_owner is not None:
            self.players[longest_owner].bonus_vp += VP_LONGEST_ROUTE
        if largest_army_owner is not None:
            self.players[largest_army_owner].bonus_vp += VP_LARGEST_ARMY

        for player in self.players.values():
            player.update_victory_points()

    def step_roll_phase(self):
        dice_value = roll_dice()
        self.last_roll = dice_value
        self.last_robber_event = None

        if dice_value == 7:
            self.robber_pending = True
            self._apply_robber_effects()
        else:
            self.robber_pending = False
            self._distribute_resources(dice_value)

        self.phase_router.complete_roll_phase(self)

    def _distribute_resources(self, dice_value: int):
        if dice_value == 7:
            return

        payouts: Dict[Resource, Dict[PlayerId, int]] = {r: {} for r in COLLECTABLE_RESOURCES}
        demands: Dict[Resource, int] = {r: 0 for r in COLLECTABLE_RESOURCES}

        tiles = self.board.get_tiles_for_roll(dice_value)

        for tile in tiles:
            resource = tile.resource
            if resource == Resource.DESERT:
                continue

            for vertex in tile.vertices:
                if vertex.building is None:
                    continue

                owner = vertex.building.owner
                amount = 0
                if vertex.building.type == BuildingType.SETTLEMENT:
                    amount = 1
                elif vertex.building.type == BuildingType.CITY:
                    amount = 2

                if amount > 0:
                    payouts[resource][owner] = payouts[resource].get(owner, 0) + amount
                    demands[resource] += amount

        for resource, total_demand in demands.items():
            # According to Catan rules, if the bank cannot pay out the total amount of a resource
            # for a given roll, then no player receives any of that resource. This can happen if
            # the resource supply in the bank is depleted.
            if self.resource_bank.get(resource, 0) >= total_demand:
                for player_id, amount in payouts[resource].items():
                    self.players[player_id].add_resource(resource, amount)
                    self.resource_bank[resource] -= amount

    def _discard_half_random(self, player: AgentState) -> Dict[Resource, int]:
        total = sum(int(v) for v in player.resources.values())
        to_discard = total // 2
        discarded = {
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        }

        if to_discard <= 0:
            return discarded

        available_cards: List[Resource] = []
        for resource, count in player.resources.items():
            available_cards.extend([resource] * int(count))

        self.random.shuffle(available_cards)
        chosen = available_cards[:to_discard]

        for resource in chosen:
            player.resources[resource] -= 1
            discarded[resource] += 1

        return discarded

    def _steal_one_random_resource(self, victim: AgentState, thief: AgentState) -> Optional[Resource]:
        pool: List[Resource] = []
        for resource, count in victim.resources.items():
            pool.extend([resource] * int(count))

        if not pool:
            return None

        stolen = self.random.choice(pool)
        victim.resources[stolen] -= 1
        thief.resources[stolen] += 1
        return stolen

    def _apply_robber_effects(self):
        self.robber_discard_required = {}
        self.robber_discard_queue = []
        self.robber_move_pending_player = self.player_order[self.current_player_idx]

        for player_id, player in self.players.items():
            total = sum(int(v) for v in player.resources.values())
            if total > 7:
                self.robber_discard_required[player_id] = total // 2
                self.robber_discard_queue.append(player_id)

        self.last_robber_event = {
            "rolled_seven": True,
            "discarded": {},
            "robber_moved": False,
            "moved_to": None,
            "stolen_from": None,
            "stolen_resource": None,
        }

    def _move_robber(self, target_tile_id: int, victim_id: Optional[PlayerId] = None):
        if target_tile_id < 0 or target_tile_id >= len(self.board.tiles):
            return

        target_tile = self.board.tiles[target_tile_id]
        if target_tile.has_robber:
            return

        move_robber(self.board, target_tile_id)

        thief_id = self.get_current_player_id()
        thief = self.players[thief_id]

        adjacent_owners = set()
        for vertex in target_tile.vertices:
            if vertex.building and vertex.building.owner != thief_id:
                owner = vertex.building.owner
                if sum(int(v) for v in self.players[owner].resources.values()) > 0:
                    adjacent_owners.add(owner)

        stolen_from = None
        stolen_resource = None
        if adjacent_owners:
            victim_to_rob_id = None
            if victim_id is not None and victim_id in adjacent_owners:
                victim_to_rob_id = victim_id
            else:
                victim_to_rob_id = self.random.choice(list(adjacent_owners))

            victim = self.players[victim_to_rob_id]
            stolen_resource = self._steal_one_random_resource(victim, thief)
            stolen_from = victim_to_rob_id if stolen_resource is not None else None

        self.robber_pending = False
        self.robber_move_pending_player = None
        self.robber_discard_queue = []
        self.robber_discard_required = {}
        self.last_robber_event = {
            "rolled_seven": True,
            "discarded": self.last_robber_event.get("discarded", {}) if self.last_robber_event else {},
            "robber_moved": True,
            "moved_to": str(target_tile_id),
            "stolen_from": str(stolen_from) if stolen_from is not None else None,
            "stolen_resource": stolen_resource.name if stolen_resource is not None else None,
        }

    def _move_robber_for_knight(
        self, player_id: PlayerId, target_tile_id: Optional[int] = None, victim_id: Optional[PlayerId] = None
    ):
        if target_tile_id is None:
            target_tile_id = self._choose_robber_target_tile(player_id)
        if target_tile_id is None or not (0 <= target_tile_id < len(self.board.tiles)):
            return

        target_tile = self.board.tiles[target_tile_id]
        if target_tile.has_robber:
            return

        move_robber(self.board, target_tile_id)

        thief = self.players[player_id]
        adjacent_owners = set()
        for vertex in target_tile.vertices:
            if vertex.building and vertex.building.owner != player_id:
                owner = vertex.building.owner
                if sum(int(v) for v in self.players[owner].resources.values()) > 0:
                    adjacent_owners.add(owner)

        stolen_from = None
        stolen_resource = None
        if adjacent_owners:
            victim_to_rob_id = None
            if victim_id is not None and victim_id in adjacent_owners:
                victim_to_rob_id = victim_id
            else:
                victim_to_rob_id = self.random.choice(list(adjacent_owners))

            victim = self.players[victim_to_rob_id]
            stolen_resource = self._steal_one_random_resource(victim, thief)
            stolen_from = victim_to_rob_id if stolen_resource is not None else None

        self.last_robber_event = {
            "rolled_seven": False,
            "discarded": {},
            "robber_moved": True,
            "moved_to": str(target_tile_id),
            "stolen_from": str(stolen_from) if stolen_from is not None else None,
            "stolen_resource": stolen_resource.name if stolen_resource is not None else None,
        }

    def _assign_initial_settlement_resources(self, vertex_id: int, player_id: PlayerId) -> None:
        player = self.players[player_id]
        vertex = self.board.vertices[vertex_id]
        for tile in vertex.tiles:
            resource = tile.resource
            if resource != Resource.DESERT:
                if self.resource_bank.get(resource, 0) > 0:
                    player.add_resource(resource, 1)
                    self.resource_bank[resource] -= 1

    def _apply_discard_action(self, player_id: PlayerId, resources: Dict[Resource, int]) -> float:
        required = int(self.robber_discard_required.get(player_id, 0))
        if required <= 0:
            return -0.02

        total = 0
        for resource, amount in resources.items():
            if amount < 0 or int(self.players[player_id].resources.get(resource, 0)) < int(amount):
                return -0.02
            total += int(amount)

        if total != required:
            return -0.02

        player = self.players[player_id]
        for resource, amount in resources.items():
            if int(amount) > 0:
                player.remove_resource(resource, int(amount))
                self.resource_bank[resource] += int(amount)

        discard_log = {resource.name: int(amount) for resource, amount in resources.items() if int(amount) > 0}
        self.last_robber_event["discarded"][str(player_id)] = discard_log

        if self.robber_discard_queue and self.robber_discard_queue[0] == player_id:
            self.robber_discard_queue.pop(0)
        self.robber_discard_required.pop(player_id, None)
        return 0.0

    def _apply_bank_trade(self, player_id: PlayerId, give: Resource, receive: Resource) -> float:
        if give == receive:
            return -0.02

        player = self.players[player_id]
        rate = self._best_maritime_rate(player_id, give)
        if player.resources.get(give, 0) < rate:
            return -0.02
        if self.resource_bank.get(receive, 0) < 1:
            return -0.02

        before_resources = dict(player.resources)
        before_readiness = self._build_readiness_score(before_resources)
        before_diversity = sum(1 for value in before_resources.values() if int(value) > 0)

        player.resources[give] -= rate
        self.resource_bank[give] += rate
        player.resources[receive] += 1
        self.resource_bank[receive] -= 1

        after_resources = dict(player.resources)
        after_readiness = self._build_readiness_score(after_resources)
        after_diversity = sum(1 for value in after_resources.values() if int(value) > 0)

        if after_readiness > before_readiness:
            return 0.03
        if after_diversity > before_diversity:
            return 0.002
        return -0.01

    def apply_gameplay_action(self, action: Optional[dict]) -> float:
        player_id = self.get_current_player_id()
        player = self.players[player_id]

        if action is None:
            return -0.01

        action_type = action.get("type")
        reward = 0.0

        if self.phase_router.get_phase() == TurnPhase.SETUP:
            if self.initial_placement_stage == "settlement":
                if action_type != "build_settlement":
                    return -0.02

                vertex_id = action.get("vertex")
                if vertex_id is None or vertex_id not in self.get_valid_settlement_vertices(player_id, require_road=False):
                    return -0.02

                if player.n_settlements >= 5:
                    reward -= 0.15
                else:
                    player.n_settlements += 1
                    self.settlement_positions[player_id].add(vertex_id)
                    vertex = self.board.get_vertex_by_id(vertex_id)
                    building = Building(BuildingType.SETTLEMENT, player_id, vertex)
                    vertex.place_building(building)
                    player.update_victory_points()
                    reward += 0.20

                    if player.n_settlements == 2:
                        self._assign_initial_settlement_resources(vertex_id, player_id)
                        reward += 0.15

                self.initial_placement_stage = "road"
                self.phase_router.set_phase(TurnPhase.SETUP)
                self._check_winner()
                return reward

            if self.initial_placement_stage == "road":
                if action_type != "build_road":
                    return -0.02

                conn_id = action.get("connection")
                if conn_id is None or conn_id not in self.get_valid_road_connections(player_id):
                    return -0.02

                player.n_roads += 1
                player.roads.append(conn_id)
                self.road_positions[player_id].add(conn_id)
                connection = self.board.get_connection_by_id(conn_id)
                connection.build_road(player_id)
                self._update_special_awards()
                player.update_victory_points()
                reward += 0.08

                self.initial_placement_index += 1
                self.initial_placement_stage = "settlement"

                if self.initial_placement_index >= len(self.initial_placement_order):
                    self.initial_placement_phase = False
                    self.current_player_idx = 0
                    self.phase_router.begin_turn(self)
                else:
                    self.phase_router.set_phase(TurnPhase.SETUP)

                self._check_winner()
                return reward

            return -0.02

        if self.phase_router.get_phase() == TurnPhase.MAIN_ACTION and self.robber_pending:
            if self.robber_discard_queue:
                if action_type != "discard_cards":
                    return -0.02
            elif action_type != "move_robber":
                return -0.02

        if action_type == "build_settlement":
            vertex_id = action.get("vertex")
            if vertex_id is None or vertex_id not in self.get_valid_settlement_vertices(player_id):
                reward -= 0.10
            elif player.n_settlements >= 5:
                reward -= 0.15
            elif not player.can_pay_cost(COST_BUILD_SETTLEMENT):
                reward -= 0.10
            else:
                self._attempt_pay_cost(player, COST_BUILD_SETTLEMENT)
                player.n_settlements += 1
                self.settlement_positions[player_id].add(vertex_id)
                vertex = self.board.get_vertex_by_id(vertex_id)
                building = Building(BuildingType.SETTLEMENT, player_id, vertex)
                vertex.place_building(building)
                self._update_special_awards()
                player.update_victory_points()
                reward += 0.18

        elif action_type == "build_city":
            vertex_id = action.get("vertex")
            if vertex_id is None or vertex_id not in self.settlement_positions[player_id]:
                reward -= 0.12
            elif player.n_settlements <= 0 or player.n_cities >= 4:
                reward -= 0.12
            elif not player.can_pay_cost(COST_BUILD_CITY):
                reward -= 0.12
            else:
                self._attempt_pay_cost(player, COST_BUILD_CITY)
                player.n_settlements -= 1
                player.n_cities += 1
                self.settlement_positions[player_id].remove(vertex_id)
                self.city_positions[player_id].add(vertex_id)
                vertex = self.board.get_vertex_by_id(vertex_id)
                vertex.upgrade_to_city()
                self._update_special_awards()
                player.update_victory_points()
                reward += 0.40

        elif action_type == "build_road":
            conn_id = action.get("connection")
            if conn_id is None or conn_id not in self.get_valid_road_connections(player_id):
                reward -= 0.06
            elif player.n_settlements <= 0:
                reward -= 0.06
            elif player.n_roads >= 15:
                reward -= 0.06
            elif not player.can_pay_cost(COST_BUILD_ROAD):
                reward -= 0.06
            else:
                self._attempt_pay_cost(player, COST_BUILD_ROAD)
                player.n_roads += 1
                player.roads.append(conn_id)
                self.road_positions[player_id].add(conn_id)
                connection = self.board.get_connection_by_id(conn_id)
                connection.build_road(player_id)
                self._update_special_awards()
                player.update_victory_points()
                reward += 0.005

                structure_count = player.n_settlements + player.n_cities
                if structure_count > 0 and player.n_roads > (2 * structure_count):
                    reward -= 0.03

        elif action_type == "discard_cards":
            resources = action.get("resources", {})
            reward += self._apply_discard_action(player_id, resources)

        elif action_type == "buy_dev_card":
            if not player.can_pay_cost(COST_BUY_DEV_CARD) or not self.dev_card_deck:
                reward -= 0.02
            else:
                self._attempt_pay_cost(player, COST_BUY_DEV_CARD)
                card = self._draw_dev_card()
                if card is None:
                    reward -= 0.02
                else:
                    if card == DevCard.VICTORY_POINT:
                        player.hidden_vp_cards += 1
                        player.update_victory_points()
                        self._update_special_awards()
                        reward += 0.28
                    else:
                        player.add_dev_card(card, playable=False)
                        reward += 0.08

        elif action_type == "play_dev_card":
            card_value = action.get("card")
            try:
                card = DevCard(card_value) if not isinstance(card_value, DevCard) else card_value
            except (TypeError, ValueError):
                reward -= 0.02
            else:
                reward += self._play_dev_card(player_id, card, action)

        elif action_type == "bank_trade":
            give = action.get("give")
            receive = action.get("receive")
            if give is None or receive is None:
                reward -= 0.02
            else:
                reward += self._apply_bank_trade(player_id, give, receive)

        elif action_type == "move_robber":
            if not self.robber_pending:
                reward -= 0.02
            else:
                tile_id = action.get("tile")
                victim_id = action.get("victim")
                if tile_id is None or not isinstance(tile_id, int) or not (0 <= tile_id < len(self.board.tiles)):
                    reward -= 0.02
                elif self.board.tiles[tile_id].has_robber:
                    reward -= 0.02
                else:
                    self._move_robber(tile_id, victim_id=victim_id)
                    reward += 0.03

        elif action_type == "end_turn":
            self.phase_router.complete_end_turn_phase(self)
            self.next_player()
            self.phase_router.begin_turn(self)
            return reward

        elif action_type == "end_main_action":
            self._advance_after_main_action()
            if self._can_take_high_value_action(player_id):
                return reward - 0.02
            return reward - 0.005

        else:
            return -0.02

        self._check_winner()

        return reward

    def apply_trade_proposal(self, action: Optional[dict]) -> float:
        if not self.enable_trading:
            self.phase_router.set_phase(TurnPhase.END_TURN)
            return 0.0

        reward = 0.0

        if action is None:
            self.phase_router.complete_trade_propose_phase(self)
            return reward

        action_type = action.get("type")

        if action_type == "skip_trade":
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.004

        if action_type != "propose_trade":
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.01

        player_id = self.get_current_player_id()
        target = action.get("target")
        offer = action.get("offer")
        request = action.get("request")

        if target is None or offer is None or request is None:
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.01

        offer_value = self._trade_value(offer)
        request_value = self._trade_value(request)

        if offer_value <= 0 or request_value <= 0:
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.01

        success = self.trade_manager.submit_trade(
            players=self.players,
            proposer=player_id,
            target=target,
            offer=offer,
            request=request,
            turn_number=self.turn_number,
            phase_index=self.phase_router.phase_index,
        )

        if success:
            reward += 0.006
            self.phase_router.set_phase(TurnPhase.TRADE_RESPOND)
        else:
            reward -= 0.01
            self.phase_router.complete_trade_propose_phase(self)

        return reward

    def apply_trade_response(self, action: Optional[dict]) -> tuple[float, dict]:
        if not self.enable_trading:
            self.phase_router.set_phase(TurnPhase.END_TURN)
            return 0.0

        reward = 0.0

        pending = self.trade_manager.get_pending_trade()
        if pending is None:
            self.phase_router.complete_trade_respond_phase(self)
            return reward, {}

        response_player = pending.target

        if action is None:
            response = TradeResponse(response_type="reject")
        else:
            response = TradeResponse(
                response_type=action.get("response_type", "reject"),
                counter_offer=action.get("counter_offer"),
                counter_request=action.get("counter_request"),
            )

        trade_info = {}
        responded = self.trade_manager.respond_to_trade(
            players=self.players,
            response_player=response_player,
            response=response,
        )

        if response.response_type == "accept" and responded:
            trade_info = {"trade_details": pending.to_dict()}
            reward += 0.02
        elif response.response_type == "counter" and responded:
            trade_info = {"trade_details": pending.to_dict()}
            reward += 0.004
        elif response.response_type == "reject":
            trade_info = {"trade_details": pending.to_dict()}
            reward -= 0.003
        else:
            reward -= 0.008

        for player in self.players.values():
            player.update_victory_points()

        self._check_winner()
        self.phase_router.complete_trade_respond_phase(self)

        return reward, trade_info

    def _check_winner(self):
        for player_id, player in self.players.items():
            player.update_victory_points()
            if player.victory_points >= VICTORY_POINTS_TARGET:
                self.winner = player_id
                return

    def _serialize_port(self, port) -> dict:
        return {
            "id": port.id,
            "resource": port.resource.name if port.resource else "GENERIC",
            "exchange_rate": port.exchange_rate,
            "vertices": [v.id for v in port.vertices],
        }

    def _serialize_tile(self, tile) -> dict:
        return {
            "id": tile.id,
            "resource": tile.resource,
            "number": tile.number,
            "has_robber": tile.has_robber,
        }

    def get_observation(self) -> dict:
        current_player = self.get_current_player_id()

        for player in self.players.values():
            player.update_victory_points()

        return {
            "game": {
                "turn_number": self.turn_number,
                "current_player": current_player,
                "phase": self.phase_router.get_phase(),
                "enable_trading": self.enable_trading,
                "initial_placement_phase": self.initial_placement_phase,
                "initial_placement_index": self.initial_placement_index,
                "initial_placement_stage": self.initial_placement_stage,
                "last_roll": self.last_roll,
                "robber_pending": self.robber_pending,
                "last_robber_event": self.last_robber_event,
                "dev_card_deck_size": len(self.dev_card_deck),
                "longest_road_owner": self.longest_road_owner,
                "largest_army_owner": self.largest_army_owner,
                "resource_bank": {r.name: v for r, v in self.resource_bank.items()},
            },
            "board": {
                "tiles": [self._serialize_tile(tile) for tile in self.board.tiles],
                "ports": [self._serialize_port(port) for port in self.board.ports],
            },
            "player": self.players[current_player].as_dict(private=True),
            "players": {
                pid: self.players[pid].as_dict(private=False)
                for pid in self.players
            },
            "trade": self.trade_manager.get_pending_trade(),
            "trade_history": self.trade_history.build_sequence_tensor_dict(),
        }

    def step(self, action: Optional[dict]):
        if self.winner is not None:
            return self.get_observation(), 0.0, True, {}

        acting_player_id = self.get_current_player_id()
        pre_step_player = self.players[acting_player_id]
        pre_step_stats = {
            "vp": float(pre_step_player.update_victory_points()),
            "resources": {k: int(v) for k, v in pre_step_player.resources.items()},
        }

        phase = self.phase_router.get_phase()
        reward = 0.0
        trade_info = {}

        if phase == TurnPhase.SETUP:
            reward = self.apply_gameplay_action(action)
        elif phase == TurnPhase.ROLL:
            self.step_roll_phase()
        elif phase == TurnPhase.MAIN_ACTION:
            reward = self.apply_gameplay_action(action)
        elif phase == TurnPhase.TRADE_PROPOSE:
            reward = self.apply_trade_proposal(action)
        elif phase == TurnPhase.TRADE_RESPOND:
            reward, trade_info = self.apply_trade_response(action)
        elif phase == TurnPhase.END_TURN:
            reward = self.apply_gameplay_action({"type": "end_turn"})

        obs = self.get_observation()
        done = self.winner is not None

        post_step_player = self.players[acting_player_id]
        post_step_stats = {
            "vp": float(post_step_player.update_victory_points()),
            "resources": {k: int(v) for k, v in post_step_player.resources.items()},
        }
        info = {
            "acting_player_id": acting_player_id,
            "pre_step_stats": pre_step_stats,
            "post_step_stats": post_step_stats,
            "winner": self.winner,
        }
        if trade_info:
            info.update(trade_info)

        if done:
            if self.winner == acting_player_id:
                reward += 1.0
            else:
                reward -= 1.0

        return obs, reward, done, info
