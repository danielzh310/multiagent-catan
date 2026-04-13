from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from core.constants import PlayerId, Resource
from core.trade_history import TradeEvent, TradeHistory, normalize_trade_vector

MAX_COUNTER_TRADES = 3


@dataclass
class TradeProposal:
    proposer: PlayerId
    target: PlayerId
    offer: Dict[Resource, int]
    request: Dict[Resource, int]
    turn_number: int
    phase_index: int
    counter_count: int = 0

    def to_dict(self) -> dict:
        return {
            "proposer": self.proposer,
            "target": self.target,
            "offer": dict(self.offer),
            "request": dict(self.request),
            "turn_number": self.turn_number,
            "phase_index": self.phase_index,
            "counter_count": self.counter_count,
        }



@dataclass
class TradeResponse:
    response_type: str
    counter_offer: Optional[Dict[Resource, int]] = None
    counter_request: Optional[Dict[Resource, int]] = None


class TradeManager:
    def __init__(self, trade_history: Optional[TradeHistory] = None):
        self.trade_history = trade_history if trade_history is not None else TradeHistory()
        self.pending_trade: Optional[TradeProposal] = None

    def reset(self) -> None:
        self.pending_trade = None
        self.trade_history.reset()

    def has_pending_trade(self) -> bool:
        return self.pending_trade is not None

    def can_player_afford(self, player_state, trade_vector: Dict[Resource, int]) -> bool:
        for resource, amount in trade_vector.items():
            if amount < 0:
                return False
            if player_state.resources.get(resource, 0) < amount:
                return False
        return True

    def validate_trade_vector(self, trade_vector: Dict[Resource, int]) -> bool:
        total = 0
        for resource, amount in trade_vector.items():
            if resource == Resource.DESERT:
                return False
            if amount < 0:
                return False
            total += amount
        return total > 0

    def validate_trade_proposal(self, players, proposal: TradeProposal) -> bool:
        if proposal.proposer == proposal.target:
            return False

        proposer_state = players[proposal.proposer]

        offer = normalize_trade_vector(proposal.offer)
        request = normalize_trade_vector(proposal.request)

        if not self.validate_trade_vector(offer):
            return False
        if not self.validate_trade_vector(request):
            return False
        if not self.can_player_afford(proposer_state, offer):
            return False

        return True

    def submit_trade(
        self,
        players,
        proposer: PlayerId,
        target: PlayerId,
        offer: Dict[Resource, int],
        request: Dict[Resource, int],
        turn_number: int,
        phase_index: int,
    ) -> bool:
        if self.pending_trade is not None:
            return False

        proposal = TradeProposal(
            proposer=proposer,
            target=target,
            offer=normalize_trade_vector(offer),
            request=normalize_trade_vector(request),
            turn_number=turn_number,
            phase_index=phase_index,
        )

        if not self.validate_trade_proposal(players, proposal):
            return False

        self.pending_trade = proposal
        return True

    def build_trade_response_targets(self, players, target_player: PlayerId) -> dict:
        if self.pending_trade is None:
            return {
                "has_pending_trade": False,
                "target_player": target_player,
                "offer": normalize_trade_vector(None),
                "request": normalize_trade_vector(None),
            }

        return {
            "has_pending_trade": True,
            "target_player": target_player,
            "proposer": self.pending_trade.proposer,
            "offer": dict(self.pending_trade.offer),
            "request": dict(self.pending_trade.request),
        }

    def respond_to_trade(
        self,
        players,
        response_player: PlayerId,
        response: TradeResponse,
    ) -> bool:
        if self.pending_trade is None:
            return False

        proposal = self.pending_trade

        if response_player != proposal.target:
            return False

        if response.response_type == "reject":
            self._record_trade_event(players, proposal, response, accepted=False, executed=False)
            self.pending_trade = None
            return True

        if response.response_type == "accept":
            proposer_state = players[proposal.proposer]
            target_state = players[proposal.target]

            if not self.can_player_afford(proposer_state, proposal.offer):
                self._record_trade_event(players, proposal, response, accepted=False, executed=False)
                self.pending_trade = None
                return False

            if not self.can_player_afford(target_state, proposal.request):
                self._record_trade_event(players, proposal, response, accepted=False, executed=False)
                self.pending_trade = None
                return False

            self._execute_trade(players, proposal.offer, proposal.request, proposal.proposer, proposal.target)
            proposer_state.update_victory_points()
            target_state.update_victory_points()

            self._record_trade_event(players, proposal, response, accepted=True, executed=True)
            self.pending_trade = None
            return True

        if response.response_type == "counter":
            if proposal.counter_count >= MAX_COUNTER_TRADES:
                self._record_trade_event(players, proposal, response, accepted=False, executed=False)
                self.pending_trade = None
                return False

            counter_offer = normalize_trade_vector(response.counter_offer)
            counter_request = normalize_trade_vector(response.counter_request)

            if not self.validate_trade_vector(counter_offer):
                return False
            if not self.validate_trade_vector(counter_request):
                return False

            responder_state = players[proposal.target]
            if not self.can_player_afford(responder_state, counter_offer):
                return False

            self._record_trade_event(players, proposal, response, accepted=False, executed=False)

            self.pending_trade = TradeProposal(
                proposer=proposal.target,
                target=proposal.proposer,
                offer=counter_offer,
                request=counter_request,
                turn_number=proposal.turn_number,
                phase_index=proposal.phase_index,
                counter_count=proposal.counter_count + 1,
            )
            return True

        return False

    def _execute_trade(
        self,
        players,
        offer: Dict[Resource, int],
        request: Dict[Resource, int],
        proposer: PlayerId,
        target: PlayerId,
    ) -> None:
        proposer_state = players[proposer]
        target_state = players[target]

        for resource, amount in offer.items():
            proposer_state.resources[resource] -= amount
            target_state.resources[resource] += amount

        for resource, amount in request.items():
            target_state.resources[resource] -= amount
            proposer_state.resources[resource] += amount

    def _record_trade_event(
        self,
        players,
        proposal: TradeProposal,
        response: TradeResponse,
        accepted: bool,
        executed: bool,
    ) -> None:
        proposer_state = players[proposal.proposer]
        target_state = players[proposal.target]

        proposer_state.update_victory_points()
        target_state.update_victory_points()

        event = TradeEvent(
            turn_number=proposal.turn_number,
            phase_index=proposal.phase_index,
            proposer=proposal.proposer,
            target=proposal.target,
            offer=dict(proposal.offer),
            request=dict(proposal.request),
            response_type=response.response_type,
            counter_offer=normalize_trade_vector(response.counter_offer),
            counter_request=normalize_trade_vector(response.counter_request),
            accepted=accepted,
            executed=executed,
            proposer_vp_before=int(proposer_state.victory_points),
            target_vp_before=int(target_state.victory_points),
            proposer_vp_after=int(proposer_state.victory_points),
            target_vp_after=int(target_state.victory_points),
        )

        self.trade_history.add_event(event)

    def clear_pending_trade(self) -> None:
        self.pending_trade = None

    def get_pending_trade(self) -> Optional[TradeProposal]:
        return self.pending_trade

    def get_trade_history(self) -> TradeHistory:
        return self.trade_history

    def legal_trade_targets(self, current_player: PlayerId, all_players: List[PlayerId]) -> List[PlayerId]:
        return [player_id for player_id in all_players if player_id != current_player]