from __future__ import annotations

import random
from typing import Dict, List, Optional

from core.constants import PlayerId, VICTORY_POINTS_TARGET, Resource
from core.agent_state import AgentState
from core.helpers import roll_dice
from core.trade_manager import TradeManager, TradeResponse
from core.trade_history import TradeHistory
from core.phase_router import PhaseRouter, TurnPhase


class CatanEngine:
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

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
        self.robber_pending = False

        self.trade_history = TradeHistory()
        self.trade_manager = TradeManager(self.trade_history)
        self.phase_router = PhaseRouter()

        self.winner: Optional[PlayerId] = None

    def reset(self):
        for player in self.players.values():
            player.resources = {r: 0 for r in player.resources}
            player.n_settlements = 0
            player.n_cities = 0
            player.n_roads = 0
            player.bonus_vp = 0
            player.revealed_vp_cards = 0

        self.current_player_idx = 0
        self.turn_number = 0

        self.initial_placement_phase = False
        self.robber_pending = False

        self.trade_manager.reset()
        self.phase_router.reset()

        self.winner = None

    def get_current_player_id(self) -> PlayerId:
        return self.player_order[self.current_player_idx]

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.player_order)
        if self.current_player_idx == 0:
            self.turn_number += 1

    def step_roll_phase(self):
        dice_value = roll_dice()

        if dice_value == 7:
            self.robber_pending = True
        else:
            self._distribute_resources(dice_value)

        self.phase_router.complete_roll_phase(self)

    def _distribute_resources(self, dice_value: int):
        for player in self.players.values():
            for resource in player.resources:
                player.resources[resource] += self.random.randint(0, 1)

    def _resource_total(self, player_id: PlayerId) -> int:
        player = self.players[player_id]
        return sum(int(v) for v in player.resources.values())

    def _resource_diversity(self, player_id: PlayerId) -> int:
        player = self.players[player_id]
        return sum(1 for v in player.resources.values() if int(v) > 0)

    def _trade_value(self, trade_vector: Dict[Resource, int]) -> int:
        return sum(int(v) for v in trade_vector.values())

    def _vp(self, player_id: PlayerId) -> int:
        return int(self.players[player_id].total_vp())

    def apply_gameplay_action(self, action: dict) -> float:
        player_id = self.get_current_player_id()
        player = self.players[player_id]

        action_type = action.get("type")
        reward = 0.0

        before_vp = self._vp(player_id)
        before_diversity = self._resource_diversity(player_id)

        if action_type == "build_settlement":
            player.n_settlements += 1
            player.bonus_vp += 1
            reward += 0.25

        elif action_type == "build_city":
            if player.n_settlements > 0:
                player.n_settlements -= 1
                player.n_cities += 1
                player.bonus_vp += 1
                reward += 0.35
            else:
                reward -= 0.02

        elif action_type == "build_road":
            player.n_roads += 1
            reward += 0.05

        elif action_type == "end_turn":
            self.phase_router.complete_end_turn_phase(self)
            self.next_player()
            self.phase_router.begin_turn(self)
            return reward

        elif action_type == "end_main_action":
            self.phase_router.complete_main_action_phase(self)
            return reward - 0.005

        self._check_winner()

        after_vp = self._vp(player_id)
        after_diversity = self._resource_diversity(player_id)

        reward += 0.20 * max(after_vp - before_vp, 0)
        reward += 0.01 * max(after_diversity - before_diversity, 0)

        if self.phase_router.get_phase() == TurnPhase.MAIN_ACTION:
            self.phase_router.complete_main_action_phase(self)

        return reward

    def apply_trade_proposal(self, action: Optional[dict]) -> float:
        reward = 0.0

        if action is None:
            self.phase_router.complete_trade_propose_phase(self)
            return reward

        action_type = action.get("type")

        if action_type == "skip_trade":
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.003

        if action_type != "propose_trade":
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.01

        player_id = self.get_current_player_id()

        target = action.get("target")
        offer = action.get("offer")
        request = action.get("request")

        if target is None or offer is None or request is None:
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.02

        offer_value = self._trade_value(offer)
        request_value = self._trade_value(request)

        if offer_value <= 0 or request_value <= 0:
            self.phase_router.complete_trade_propose_phase(self)
            return reward - 0.02

        diversity_bonus = 0.0
        proposer_div_before = self._resource_diversity(player_id)
        requested_types = sum(1 for v in request.values() if int(v) > 0)
        if proposer_div_before < 3:
            diversity_bonus += 0.01 * requested_types

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
            reward += 0.015
            reward += 0.004 * min(request_value, 3)
            reward += diversity_bonus
            self.phase_router.set_phase(TurnPhase.TRADE_RESPOND)
        else:
            reward -= 0.02
            self.phase_router.complete_trade_propose_phase(self)

        return reward

    def apply_trade_response(self, action: Optional[dict]) -> float:
        reward = 0.0

        pending = self.trade_manager.get_pending_trade()
        if pending is None:
            self.phase_router.complete_trade_respond_phase(self)
            return reward

        response_player = pending.target
        proposer = pending.proposer

        before_total = self._resource_total(response_player)
        before_diversity = self._resource_diversity(response_player)
        before_vp = self._vp(response_player)

        proposer_before_total = self._resource_total(proposer)
        proposer_before_diversity = self._resource_diversity(proposer)
        proposer_before_vp = self._vp(proposer)

        if action is None:
            response = TradeResponse(response_type="reject")
        else:
            response = TradeResponse(
                response_type=action.get("response_type", "reject"),
                counter_offer=action.get("counter_offer"),
                counter_request=action.get("counter_request"),
            )

        responded = self.trade_manager.respond_to_trade(
            players=self.players,
            response_player=response_player,
            response=response,
        )

        after_total = self._resource_total(response_player)
        after_diversity = self._resource_diversity(response_player)
        after_vp = self._vp(response_player)

        proposer_after_total = self._resource_total(proposer)
        proposer_after_diversity = self._resource_diversity(proposer)
        proposer_after_vp = self._vp(proposer)

        if response.response_type == "accept" and responded:
            reward += 0.08
            reward += 0.015 * max(after_diversity - before_diversity, 0)
            reward += 0.008 * max(after_total - before_total, 0)
            reward += 0.08 * max(after_vp - before_vp, 0)

            proposer_gain = 0.0
            proposer_gain += 0.015 * max(proposer_after_diversity - proposer_before_diversity, 0)
            proposer_gain += 0.008 * max(proposer_after_total - proposer_before_total, 0)
            proposer_gain += 0.08 * max(proposer_after_vp - proposer_before_vp, 0)

            reward += 0.25 * proposer_gain

        elif response.response_type == "counter" and responded:
            reward += 0.03

        elif response.response_type == "reject":
            reward -= 0.003

        else:
            reward -= 0.01

        self._check_winner()
        self.phase_router.complete_trade_respond_phase(self)

        return reward

    def _check_winner(self):
        for player_id, player in self.players.items():
            if player.total_vp() >= VICTORY_POINTS_TARGET:
                self.winner = player_id
                return

    def get_observation(self) -> dict:
        current_player = self.get_current_player_id()

        return {
            "game": {
                "turn_number": self.turn_number,
                "current_player": current_player,
                "phase": self.phase_router.get_phase(),
            },
            "player": self.players[current_player].as_dict(),
            "players": {
                pid: self.players[pid].as_dict()
                for pid in self.players
            },
            "trade": self.trade_manager.get_pending_trade(),
            "trade_history": self.trade_history.build_sequence_tensor_dict(),
        }

    def step(self, action: Optional[dict]):
        if self.winner is not None:
            return self.get_observation(), 0.0, True, {}

        phase = self.phase_router.get_phase()
        reward = 0.0

        if phase == TurnPhase.ROLL:
            self.step_roll_phase()

        elif phase == TurnPhase.MAIN_ACTION:
            reward = self.apply_gameplay_action(action)

        elif phase == TurnPhase.TRADE_PROPOSE:
            reward = self.apply_trade_proposal(action)

        elif phase == TurnPhase.TRADE_RESPOND:
            reward = self.apply_trade_response(action)

        elif phase == TurnPhase.END_TURN:
            reward = self.apply_gameplay_action({"type": "end_turn"})

        obs = self.get_observation()
        done = self.winner is not None

        if done:
            winner = self.winner
            if winner == self.get_current_player_id():
                reward += 1.0
            else:
                reward -= 1.0

        return obs, reward, done, {}