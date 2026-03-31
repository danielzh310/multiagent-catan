from __future__ import annotations

from typing import List, Optional

from core.constants import PlayerId, Resource, DevCard, COST_BUILD_ROAD, COST_BUILD_SETTLEMENT, COST_BUILD_CITY, COST_BUY_DEV_CARD, cost_to_dict
from core.engine import CatanEngine
from core.phase_router import ControllerType, TurnPhase


class CatanEnv:
    MAX_DISCARD_ACTIONS = 64

    def __init__(self, seed: Optional[int] = None, enable_trading: bool = True):
        self.engine = CatanEngine(seed=seed, enable_trading=enable_trading)

    def reset(self) -> dict:
        self.engine.reset()
        return self.get_observation()

    def step(self, action: Optional[dict]):
        obs, reward, done, info = self.engine.step(action)

        decision = self.engine.phase_router.get_controller(self.engine)
        obs["controller"] = {
            "phase": decision.phase,
            "controller": decision.controller,
            "acting_player": decision.acting_player,
            "target_player": decision.target_player,
        }
        obs["legal_actions"] = self.get_legal_actions()

        return obs, reward, done, info

    def get_observation(self) -> dict:
        obs = self.engine.get_observation()
        decision = self.engine.phase_router.get_controller(self.engine)

        obs["controller"] = {
            "phase": decision.phase,
            "controller": decision.controller,
            "acting_player": decision.acting_player,
            "target_player": decision.target_player,
        }

        obs["legal_actions"] = self.get_legal_actions()
        return obs

    def get_current_player_id(self) -> PlayerId:
        return self.engine.get_current_player_id()

    def get_phase(self) -> TurnPhase:
        return self.engine.phase_router.get_phase()

    def get_controller_type(self) -> ControllerType:
        decision = self.engine.phase_router.get_controller(self.engine)
        return decision.controller

    def get_trade_history(self) -> dict:
        return self.engine.trade_history.build_sequence_tensor_dict()

    def get_pending_trade(self):
        return self.engine.trade_manager.get_pending_trade()

    def get_last_roll(self):
        return self.engine.last_roll

    def get_last_robber_event(self):
        return self.engine.last_robber_event

    def get_legal_actions(self) -> List[dict]:
        phase = self.engine.phase_router.get_phase()
        current_player = self.engine.get_current_player_id()

        if self.engine.winner is not None:
            return []

        if phase == TurnPhase.SETUP:
            player_id = current_player
            if self.engine.initial_placement_stage == "settlement":
                valid_settlements = self.engine.get_valid_settlement_vertices(player_id, require_road=False)
                if valid_settlements:
                    return [{"type": "build_settlement", "vertex": v} for v in valid_settlements]
                return []
            else:
                valid_roads = self.engine.get_valid_road_connections(player_id)
                if valid_roads:
                    return [{"type": "build_road", "connection": c} for c in valid_roads]
                return []

        if phase == TurnPhase.ROLL:
            return [{"type": "roll"}]

        if phase == TurnPhase.MAIN_ACTION:
            if self.engine.robber_pending:
                if self.engine.robber_discard_queue:
                    return self._get_legal_discard_actions(current_player)
                current_robber_tile = next((t.id for t in self.engine.board.tiles if t.has_robber), None)
                moves = []
                for t in self.engine.board.tiles:
                    if t.id != current_robber_tile:
                        moves.append({"type": "move_robber", "tile": t.id})
                return moves
            return self._get_legal_gameplay_actions(current_player)

        if phase == TurnPhase.TRADE_PROPOSE:
            if not self.engine.enable_trading:
                return [{"type": "skip_trade"}]
            return self._get_legal_trade_proposals(current_player)

        if phase == TurnPhase.TRADE_RESPOND:
            if not self.engine.enable_trading:
                return [{"type": "reject_trade", "response_type": "reject"}]
            return self._get_legal_trade_responses(current_player)

        if phase == TurnPhase.END_TURN:
            return [{"type": "end_turn"}]

        return []

    def _get_legal_gameplay_actions(self, player_id: PlayerId) -> List[dict]:
        player = self.engine.players[player_id]
        actions = []

        valid_settlements = self.engine.get_valid_settlement_vertices(player_id)
        if (
            player.n_settlements < 5
            and player.can_pay_cost(COST_BUILD_SETTLEMENT)
            and valid_settlements
        ):
            for vertex_id in valid_settlements:
                actions.append({"type": "build_settlement", "vertex": vertex_id})

        valid_roads = self.engine.get_valid_road_connections(player_id)
        if (
            player.n_settlements > 0
            and player.n_roads < 15
            and player.can_pay_cost(COST_BUILD_ROAD)
            and valid_roads
        ):
            for conn_id in valid_roads:
                actions.append({"type": "build_road", "connection": conn_id})

        if player.n_settlements > 0 and player.n_cities < 4 and player.can_pay_cost(COST_BUILD_CITY):
            for vertex_id in self.engine.settlement_positions[player_id]:
                actions.append({"type": "build_city", "vertex": vertex_id})

        if player.can_pay_cost(COST_BUY_DEV_CARD) and self.engine.dev_card_deck:
            actions.append({"type": "buy_dev_card"})

        if player.can_play_dev_card(DevCard.KNIGHT):
            current_robber_tile = next((t.id for t in self.engine.board.tiles if t.has_robber), None)
            for tile in self.engine.board.tiles:
                if tile.id != current_robber_tile:
                    actions.append({"type": "play_dev_card", "card": int(DevCard.KNIGHT), "tile": tile.id})
        if player.can_play_dev_card(DevCard.ROAD_BUILDING):
            actions.extend(self._get_road_building_actions(player_id))
        if player.can_play_dev_card(DevCard.INVENTION):
            resources = [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]
            for resource_1 in resources:
                for resource_2 in resources:
                    actions.append({
                        "type": "play_dev_card",
                        "card": int(DevCard.INVENTION),
                        "resource_1": int(resource_1),
                        "resource_2": int(resource_2),
                    })
        if player.can_play_dev_card(DevCard.MONOPOLY):
            for resource in (Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE):
                actions.append({
                    "type": "play_dev_card",
                    "card": int(DevCard.MONOPOLY),
                    "resource": int(resource),
                })

        actions.extend(self._get_legal_bank_trades(player_id))
        actions.append({"type": "end_main_action"})
        return actions

    def _get_legal_bank_trades(self, player_id: PlayerId) -> List[dict]:
        player = self.engine.players[player_id]
        actions = []

        resources = [
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ]

        for give in resources:
            rate = self.engine._best_maritime_rate(player_id, give)
            if player.resources.get(give, 0) < rate:
                continue
            for receive in resources:
                if give == receive:
                    continue
                actions.append({
                    "type": "bank_trade",
                    "give": give,
                    "receive": receive,
                    "rate": rate,
                })

        return actions

    def _get_road_building_actions(self, player_id: PlayerId) -> List[dict]:
        actions = []
        first_roads = self.engine.get_valid_road_connections(player_id)
        if not first_roads:
            return actions

        seen = set()
        for connection_1 in first_roads:
            action_single = {
                "type": "play_dev_card",
                "card": int(DevCard.ROAD_BUILDING),
                "connection_1": connection_1,
                "connection_2": None,
            }
            key_single = (connection_1, None)
            if key_single not in seen:
                actions.append(action_single)
                seen.add(key_single)

            simulated_roads = {
                pid: set(roads)
                for pid, roads in self.engine.road_positions.items()
            }
            simulated_roads[player_id].add(connection_1)

            second_roads = self.engine.board.get_valid_road_connections(
                player_id,
                self.engine.settlement_positions,
                simulated_roads,
                city_positions=self.engine.city_positions,
            )

            for connection_2 in second_roads:
                if connection_2 == connection_1:
                    continue
                key_double = (connection_1, connection_2)
                if key_double in seen:
                    continue
                actions.append({
                    "type": "play_dev_card",
                    "card": int(DevCard.ROAD_BUILDING),
                    "connection_1": connection_1,
                    "connection_2": connection_2,
                })
                seen.add(key_double)

        return actions

    def _get_legal_discard_actions(self, player_id: PlayerId) -> List[dict]:
        required = int(self.engine.robber_discard_required.get(player_id, 0))
        if required <= 0:
            return []

        player = self.engine.players[player_id]
        resources = [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]
        available = {resource: int(player.resources.get(resource, 0)) for resource in resources}
        combinations: List[dict] = []

        def build_combo(index: int, remaining: int, current: dict) -> None:
            if len(combinations) >= self.MAX_DISCARD_ACTIONS:
                return
            if index == len(resources):
                if remaining == 0:
                    combinations.append({
                        "type": "discard_cards",
                        "required": required,
                        "available": {resource.name: available[resource] for resource in resources},
                        "resources": dict(current),
                    })
                return

            resource = resources[index]
            max_take = min(available[resource], remaining)
            for amount in range(max_take, -1, -1):
                current[resource] = amount
                build_combo(index + 1, remaining - amount, current)
                if len(combinations) >= self.MAX_DISCARD_ACTIONS:
                    return
            current.pop(resource, None)

        build_combo(0, required, {})

        if combinations:
            return combinations

        return [{
            "type": "discard_cards",
            "required": required,
            "available": {resource.name: available[resource] for resource in resources},
        }]

    def _can_afford_resources(self, resources: dict, cost: tuple) -> bool:
        cost_dict = cost_to_dict(cost)
        for resource, amount in cost_dict.items():
            if resources.get(resource, 0) < amount:
                return False
        return True

    def _get_legal_trade_proposals(self, player_id: PlayerId) -> List[dict]:
        actions = [{"type": "skip_trade"}]

        player_ids = list(self.engine.players.keys())
        targets = self.engine.trade_manager.legal_trade_targets(player_id, player_ids)

        trade_templates = self._default_trade_templates()

        for target in targets:
            for offer, request in trade_templates:
                proposer_state = self.engine.players[player_id]
                if self.engine.trade_manager.can_player_afford(proposer_state, offer):
                    actions.append({
                        "type": "propose_trade",
                        "target": target,
                        "offer": offer,
                        "request": request,
                    })

        return actions

    def _get_legal_trade_responses(self, player_id: PlayerId) -> List[dict]:
        pending = self.engine.trade_manager.get_pending_trade()
        if pending is None:
            return []

        actions = [
            {"type": "reject_trade", "response_type": "reject"},
            {"type": "accept_trade", "response_type": "accept"},
        ]

        responder_state = self.engine.players[player_id]
        response_templates = self._default_trade_templates()

        for counter_offer, counter_request in response_templates:
            if self.engine.trade_manager.can_player_afford(responder_state, counter_offer):
                actions.append({
                    "type": "counter_trade",
                    "response_type": "counter",
                    "counter_offer": counter_offer,
                    "counter_request": counter_request,
                })

        return actions

    def _default_trade_templates(self):
        singles = [
            {Resource.WOOD: 1, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 1, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 1, Resource.WHEAT: 0, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 1, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 1},
        ]

        templates = []
        for offer in singles:
            for request in singles:
                if offer != request:
                    templates.append((offer, request))

        return templates

    def build_gameplay_observation(self) -> dict:
        obs = self.get_observation()
        current_player = self.get_current_player_id()

        return {
            "turn_number": obs["game"]["turn_number"],
            "phase": obs["game"]["phase"],
            "current_player": current_player,
            "initial_placement_phase": obs["game"].get("initial_placement_phase", False),
            "initial_placement_index": obs["game"].get("initial_placement_index", 0),
            "initial_placement_stage": obs["game"].get("initial_placement_stage"),
            "last_roll": obs["game"].get("last_roll"),
            "robber_pending": obs["game"].get("robber_pending", False),
            "last_robber_event": obs["game"].get("last_robber_event"),
            "self_state": obs["player"],
            "all_players": obs["players"],
            "trade_history": obs["trade_history"],
            "legal_actions": obs["legal_actions"],
        }

    def build_trade_observation(self) -> dict:
        obs = self.get_observation()
        current_player = self.get_current_player_id()

        return {
            "turn_number": obs["game"]["turn_number"],
            "phase": obs["game"]["phase"],
            "current_player": current_player,
            "initial_placement_phase": obs["game"].get("initial_placement_phase", False),
            "initial_placement_index": obs["game"].get("initial_placement_index", 0),
            "initial_placement_stage": obs["game"].get("initial_placement_stage"),
            "last_roll": obs["game"].get("last_roll"),
            "robber_pending": obs["game"].get("robber_pending", False),
            "last_robber_event": obs["game"].get("last_robber_event"),
            "self_state": obs["player"],
            "all_players": obs["players"],
            "pending_trade": obs["trade"],
            "trade_history": obs["trade_history"],
            "legal_actions": obs["legal_actions"],
        }

    def get_active_model_name(self) -> str:
        controller = self.get_controller_type()

        if controller == ControllerType.GAMEPLAY:
            return "gameplay"
        if controller == ControllerType.TRADE:
            return "trade"
        return "none"
