from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

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
    """
    Simplified but much more faithful Catan engine.

    Key improvements over the earlier version:
    - stores the most recent dice roll so logs can show it
    - applies robber consequences when 7 is rolled
    - makes resource production sparse instead of overly smooth
    - updates VP/resource state explicitly after every state mutation
    """

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
        self.initial_placement_order = [
            PlayerId.WHITE,
            PlayerId.BLUE,
            PlayerId.ORANGE,
            PlayerId.RED,
            PlayerId.RED,
            PlayerId.ORANGE,
            PlayerId.BLUE,
            PlayerId.WHITE,
        ]
        self.robber_pending = False
        self.last_roll: Optional[int] = None
        self.last_robber_event: Optional[dict] = None

        self.trade_history = TradeHistory()
        self.trade_manager = TradeManager(self.trade_history)
        self.phase_router = PhaseRouter()

        self.winner: Optional[PlayerId] = None

        # Each player gets a deterministic "production profile" over the five resources.
        # This is still simplified, but it is much better than giving everyone smooth,
        # almost guaranteed resources every turn.
        self.production_map: Dict[PlayerId, Dict[int, List[Resource]]] = {}
        self._init_production_map()

    def _init_production_map(self) -> None:
        """
        A sparse, dice-value keyed resource production map.
        This is a simplification of board geometry, but much more Catan-like than
        the previous 'always distribute based on settlement count' approach.
        """
        self.production_map = {
            PlayerId.WHITE: {
                4: [Resource.WOOD],
                5: [Resource.BRICK],
                6: [Resource.SHEEP],
                8: [Resource.WHEAT],
                9: [Resource.ORE],
                10: [Resource.WOOD],
            },
            PlayerId.BLUE: {
                4: [Resource.BRICK],
                5: [Resource.SHEEP],
                6: [Resource.WHEAT],
                8: [Resource.ORE],
                9: [Resource.WOOD],
                10: [Resource.BRICK],
            },
            PlayerId.ORANGE: {
                4: [Resource.SHEEP],
                5: [Resource.WHEAT],
                6: [Resource.WOOD],
                8: [Resource.BRICK],
                9: [Resource.ORE],
                10: [Resource.SHEEP],
            },
            PlayerId.RED: {
                4: [Resource.WHEAT],
                5: [Resource.ORE],
                6: [Resource.BRICK],
                8: [Resource.SHEEP],
                9: [Resource.WOOD],
                10: [Resource.WHEAT],
            },
        }

    def reset(self):
        for player in self.players.values():
            player.reset_for_new_game()

            # Small nonzero starting state so training can begin, but not so large
            # that the economy becomes unrealistic immediately.
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

        self.initial_placement_phase = True
        self.initial_placement_index = 0
        self.initial_placement_stage = "settlement"
        self.robber_pending = False
        self.last_roll = None
        self.last_robber_event = None

        self.trade_manager.reset()
        self.phase_router.reset()

        self.winner = None

    def get_current_player_id(self) -> PlayerId:
        if self.initial_placement_phase:
            idx = min(self.initial_placement_index, len(self.initial_placement_order)-1)
            return self.initial_placement_order[idx]
        return self.player_order[self.current_player_idx]

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.player_order)
        if self.current_player_idx == 0:
            self.turn_number += 1

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
            player.update_victory_points()
            return True
        return False

    def _advance_after_main_action(self):
        if self.enable_trading:
            self.phase_router.complete_main_action_phase(self)
        else:
            self.phase_router.set_phase(TurnPhase.END_TURN)

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
        """
        Sparse, dice-keyed production.

        Each player has a simplified production profile keyed by rolled number.
        Settlements produce 1; cities produce 2.
        No production on 7.
        """
        if dice_value == 7:
            return

        for player_id, player in self.players.items():
            if player.n_settlements <= 0 and player.n_cities <= 0:
                continue

            produced_resources = self.production_map.get(player_id, {}).get(dice_value, [])
            if not produced_resources:
                continue

            multiplier = int(player.n_settlements) + 2 * int(player.n_cities)
            if multiplier <= 0:
                continue

            for resource in produced_resources:
                player.resources[resource] += multiplier

    def _discard_half_random(self, player: AgentState) -> Dict[Resource, int]:
        """
        On a 7, players with >7 cards discard half at random.
        """
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
        """
        Simplified robber:
        - everyone with >7 cards discards half
        - current player steals one random resource from the richest opponent with cards
        """
        discarded_summary: Dict[str, Dict[str, int]] = {}

        for player_id, player in self.players.items():
            total_cards = sum(int(v) for v in player.resources.values())
            if total_cards > 7:
                discarded = self._discard_half_random(player)
                discarded_summary[str(player_id)] = {k.name: v for k, v in discarded.items() if v > 0}

        thief_id = self.get_current_player_id()
        thief = self.players[thief_id]

        candidates: List[Tuple[PlayerId, int]] = []
        for pid, player in self.players.items():
            if pid == thief_id:
                continue
            total = sum(int(v) for v in player.resources.values())
            if total > 0:
                candidates.append((pid, total))

        stolen_from = None
        stolen_resource = None

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            victim_id = candidates[0][0]
            victim = self.players[victim_id]
            stolen_resource = self._steal_one_random_resource(victim, thief)
            stolen_from = victim_id if stolen_resource is not None else None

        self.last_robber_event = {
            "rolled_seven": True,
            "discarded": discarded_summary,
            "stolen_from": str(stolen_from) if stolen_from is not None else None,
            "stolen_resource": stolen_resource.name if stolen_resource is not None else None,
        }

    def _assign_initial_settlement_resources(self, player_id: PlayerId) -> None:
        player = self.players[player_id]
        profile = self.production_map.get(player_id, {})
        resources = []
        for resource_list in profile.values():
            for r in resource_list:
                if r not in resources:
                    resources.append(r)
                if len(resources) >= 3:
                    break
            if len(resources) >= 3:
                break

        for r in resources:
            player.add_resource(r, 1)

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

                if player.n_settlements >= 5:
                    reward -= 0.15
                else:
                    player.n_settlements += 1
                    player.update_victory_points()
                    reward += 0.20

                    # second settlement draws adjacent resources (classic Catan rule)
                    if player.n_settlements == 2:
                        self._assign_initial_settlement_resources(player_id)
                        reward += 0.15

                self.initial_placement_stage = "road"
                self.phase_router.set_phase(TurnPhase.SETUP)
                self._check_winner()
                return reward

            if self.initial_placement_stage == "road":
                if action_type != "build_road":
                    return -0.02

                player.n_roads += 1
                player.update_victory_points()
                reward += 0.08

                # finished one player’s SETUP pair
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
                "last_roll": self.last_roll,
                "robber_pending": self.robber_pending,
                "last_robber_event": self.last_robber_event,
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

        if phase == TurnPhase.SETUP:
            reward = self.apply_gameplay_action(action)

        elif phase == TurnPhase.ROLL:
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