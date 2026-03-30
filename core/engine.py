from __future__ import annotations

import random
from typing import Dict, List, Optional

from core.constants import (
    PlayerId,
    VICTORY_POINTS_TARGET,
    Resource,
    COST_BUILD_ROAD,
    COST_BUILD_SETTLEMENT,
    COST_BUILD_CITY,
)
from core.agent_state import AgentState
from core.helpers import roll_dice
from core.trade_manager import TradeManager, TradeResponse
from core.trade_history import TradeHistory
from core.phase_router import PhaseRouter, TurnPhase


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
        self.robber_pending = False

        self.trade_history = TradeHistory()
        self.trade_manager = TradeManager(self.trade_history)
        self.phase_router = PhaseRouter()

        self.winner: Optional[PlayerId] = None

    def reset(self):
        for player in self.players.values():
            player.reset_for_new_game()
            # simple nonzero start so gameplay can begin, but keep it consistent
            player.resources = {
                Resource.WOOD: 2,
                Resource.BRICK: 2,
                Resource.SHEEP: 2,
                Resource.WHEAT: 2,
                Resource.ORE: 1,
            }
            player.update_victory_points()

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
            settlement_equiv = player.n_settlements + 2 * player.n_cities
            if settlement_equiv > 0:
                total_resources = max(1, (dice_value * settlement_equiv) // 6)

                resource_order = [
                    Resource.WOOD,
                    Resource.BRICK,
                    Resource.SHEEP,
                    Resource.WHEAT,
                    Resource.ORE,
                ]

                for i in range(total_resources):
                    res = resource_order[(self.turn_number + i) % len(resource_order)]
                    player.resources[res] += 1

    def _resource_total(self, player_id: PlayerId) -> int:
        return sum(int(v) for v in self.players[player_id].resources.values())

    def _resource_diversity(self, player_id: PlayerId) -> int:
        return sum(1 for v in self.players[player_id].resources.values() if int(v) > 0)

    def _trade_value(self, trade_vector: Dict[Resource, int]) -> int:
        return sum(int(v) for v in trade_vector.values())

    def _vp(self, player_id: PlayerId) -> int:
        return int(self.players[player_id].update_victory_points())

    def _attempt_pay_cost(self, player: AgentState, cost: tuple) -> bool:
        if player.can_pay_cost(cost):
            player.pay_cost(cost)
            return True
        return False

    def _advance_after_main_action(self):
        # gameplay-only debugging mode: skip trade phases entirely
        if self.enable_trading:
            self.phase_router.complete_main_action_phase(self)
        else:
            self.phase_router.set_phase(TurnPhase.END_TURN)

    def apply_gameplay_action(self, action: Optional[dict]) -> float:
        player_id = self.get_current_player_id()
        player = self.players[player_id]

        if action is None:
            return -0.01

        action_type = action.get("type")
        reward = 0.0

        before_vp = self._vp(player_id)
        before_diversity = self._resource_diversity(player_id)

        if action_type == "build_settlement":
            if player.n_settlements >= 5:
                reward -= 0.15
            elif not player.can_pay_cost(COST_BUILD_SETTLEMENT):
                reward -= 0.10
            elif player.n_settlements > 0 and player.n_roads <= 0:
                reward -= 0.10
            else:
                self._attempt_pay_cost(player, COST_BUILD_SETTLEMENT)
                player.n_settlements += 1
                player.update_victory_points()
                reward += 0.20

        elif action_type == "build_city":
            if player.n_settlements <= 0 or player.n_cities >= 4:
                reward -= 0.12
            elif not player.can_pay_cost(COST_BUILD_CITY):
                reward -= 0.12
            else:
                self._attempt_pay_cost(player, COST_BUILD_CITY)
                player.n_settlements -= 1
                player.n_cities += 1
                player.update_victory_points()
                reward += 0.24

        elif action_type == "build_road":
            if player.n_settlements <= 0:
                reward -= 0.06
            elif player.n_roads >= 15:
                reward -= 0.06
            elif not player.can_pay_cost(COST_BUILD_ROAD):
                reward -= 0.06
            else:
                self._attempt_pay_cost(player, COST_BUILD_ROAD)
                player.n_roads += 1
                player.update_victory_points()
                reward += 0.04

        elif action_type == "end_turn":
            self.phase_router.complete_end_turn_phase(self)
            self.next_player()
            self.phase_router.begin_turn(self)
            return reward

        elif action_type == "end_main_action":
            self._advance_after_main_action()
            return reward - 0.002

        else:
            return -0.02

        self._check_winner()

        after_vp = self._vp(player_id)
        after_diversity = self._resource_diversity(player_id)

        reward += 0.12 * max(after_vp - before_vp, 0)
        reward += 0.004 * max(after_diversity - before_diversity, 0)

        if self.phase_router.get_phase() == TurnPhase.MAIN_ACTION:
            self._advance_after_main_action()

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

    def apply_trade_response(self, action: Optional[dict]) -> float:
        if not self.enable_trading:
            self.phase_router.set_phase(TurnPhase.END_TURN)
            return 0.0

        reward = 0.0

        pending = self.trade_manager.get_pending_trade()
        if pending is None:
            self.phase_router.complete_trade_respond_phase(self)
            return reward

        response_player = pending.target

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

        if response.response_type == "accept" and responded:
            reward += 0.02
        elif response.response_type == "counter" and responded:
            reward += 0.004
        elif response.response_type == "reject":
            reward -= 0.003
        else:
            reward -= 0.008

        for player in self.players.values():
            player.update_victory_points()

        self._check_winner()
        self.phase_router.complete_trade_respond_phase(self)

        return reward

    def _check_winner(self):
        for player_id, player in self.players.items():
            player.update_victory_points()
            if player.victory_points >= VICTORY_POINTS_TARGET:
                self.winner = player_id
                return

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
            if self.winner == self.get_current_player_id():
                reward += 1.0
            else:
                reward -= 1.0

        return obs, reward, done, {}