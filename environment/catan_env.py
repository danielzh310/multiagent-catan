from __future__ import annotations

from typing import Dict, List, Optional

from core.constants import PlayerId, Resource, COST_BUILD_ROAD, COST_BUILD_SETTLEMENT, COST_BUILD_CITY
from core.engine import CatanEngine
from core.phase_router import ControllerType, TurnPhase


class CatanEnv:
    """
    Full-game environment with two decision streams:
    - gameplay decisions
    - trade decisions

    The environment keeps phase information explicit so the
    training loop can route control to the correct model.
    """

    def __init__(self, seed: Optional[int] = None):
        self.engine = CatanEngine(seed=seed)

    def reset(self) -> dict:
        self.engine.reset()
        self.engine.phase_router.begin_turn(self.engine)
        return self.get_observation()

    def step(self, action: Optional[dict]):
        obs, reward, done, info = self.engine.step(action)
        
        # Wrap engine observation with controller and legal_actions
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

    def get_legal_actions(self) -> List[dict]:
        phase = self.engine.phase_router.get_phase()
        current_player = self.engine.get_current_player_id()

        if self.engine.winner is not None:
            return []

        if phase == TurnPhase.ROLL:
            return [{"type": "roll"}]

        if phase == TurnPhase.MAIN_ACTION:
            return self._get_legal_gameplay_actions(current_player)

        if phase == TurnPhase.TRADE_PROPOSE:
            return self._get_legal_trade_proposals(current_player)

        if phase == TurnPhase.TRADE_RESPOND:
            return self._get_legal_trade_responses(current_player)

        if phase == TurnPhase.END_TURN:
            return [{"type": "end_turn"}]

        return []

    def _get_legal_gameplay_actions(self, player_id: PlayerId) -> List[dict]:
        player = self.engine.players[player_id]
        actions = []

        # build_road: need at least one settlement and road cap
        if player.n_settlements > 0 and player.n_roads < 15 and player.can_pay_cost(COST_BUILD_ROAD):
            actions.append({"type": "build_road"})

        # build_settlement: 5 max, 1 road after first settlement
        if player.n_settlements < 5 and player.can_pay_cost(COST_BUILD_SETTLEMENT) and (player.n_settlements == 0 or player.n_roads > 0):
            actions.append({"type": "build_settlement"})

        # build_city: convert existing settlement
        if player.n_settlements > 0 and player.n_cities < 4 and player.can_pay_cost(COST_BUILD_CITY):
            actions.append({"type": "build_city"})

        actions.append({"type": "end_main_action"})
        return actions

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

        response_templates = self._default_trade_templates()

        responder_state = self.engine.players[player_id]
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
        """
        Small starter set of trade templates.

        This will later be replaced by a richer enumerator over all legal
        trade vectors, but this is enough to wire the two-model stack.
        """
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
        """
        Observation intended for the main gameplay model.
        """
        obs = self.get_observation()
        current_player = self.get_current_player_id()

        return {
            "turn_number": obs["game"]["turn_number"],
            "phase": obs["game"]["phase"],
            "current_player": current_player,
            "self_state": obs["player"],
            "all_players": obs["players"],
            "trade_history": obs["trade_history"],
            "legal_actions": obs["legal_actions"],
        }

    def build_trade_observation(self) -> dict:
        """
        Observation intended for the trade model.
        """
        obs = self.get_observation()
        current_player = self.get_current_player_id()

        return {
            "turn_number": obs["game"]["turn_number"],
            "phase": obs["game"]["phase"],
            "current_player": current_player,
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