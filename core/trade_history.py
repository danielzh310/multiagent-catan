from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.constants import PlayerId, Resource


def empty_trade_vector() -> Dict[Resource, int]:
    return {
        Resource.WOOD: 0,
        Resource.BRICK: 0,
        Resource.SHEEP: 0,
        Resource.WHEAT: 0,
        Resource.ORE: 0,
    }


def normalize_trade_vector(vector: Optional[Dict[Resource, int]]) -> Dict[Resource, int]:
    if vector is None:
        return empty_trade_vector()

    out = empty_trade_vector()
    for resource in out:
        out[resource] = int(vector.get(resource, 0))
    return out


@dataclass
class TradeEvent:
    turn_number: int
    phase_index: int
    proposer: PlayerId
    target: PlayerId
    offer: Dict[Resource, int]
    request: Dict[Resource, int]
    response_type: str
    counter_offer: Dict[Resource, int] = field(default_factory=empty_trade_vector)
    counter_request: Dict[Resource, int] = field(default_factory=empty_trade_vector)
    accepted: bool = False
    executed: bool = False
    proposer_vp_before: int = 0
    target_vp_before: int = 0
    proposer_vp_after: int = 0
    target_vp_after: int = 0

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "phase_index": self.phase_index,
            "proposer": self.proposer,
            "target": self.target,
            "offer": dict(self.offer),
            "request": dict(self.request),
            "response_type": self.response_type,
            "counter_offer": dict(self.counter_offer),
            "counter_request": dict(self.counter_request),
            "accepted": self.accepted,
            "executed": self.executed,
            "proposer_vp_before": self.proposer_vp_before,
            "target_vp_before": self.target_vp_before,
            "proposer_vp_after": self.proposer_vp_after,
            "target_vp_after": self.target_vp_after,
        }


class TradeHistory:
    """
    Stores recent trade interactions for the full game.

    This is the source used by the trade-history encoder later.
    """

    def __init__(self, max_events: int = 64):
        self.max_events = max_events
        self.events: List[TradeEvent] = []

    def reset(self) -> None:
        self.events = []

    def add_event(self, event: TradeEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def last_event(self) -> Optional[TradeEvent]:
        if len(self.events) == 0:
            return None
        return self.events[-1]

    def get_recent_events(self, limit: Optional[int] = None) -> List[TradeEvent]:
        if limit is None or limit >= len(self.events):
            return list(self.events)
        return list(self.events[-limit:])

    def get_recent_events_for_player(self, player_id: PlayerId, limit: Optional[int] = None) -> List[TradeEvent]:
        filtered = [
            event
            for event in self.events
            if event.proposer == player_id or event.target == player_id
        ]

        if limit is None or limit >= len(filtered):
            return filtered
        return filtered[-limit:]

    def get_pair_history(
        self,
        player_a: PlayerId,
        player_b: PlayerId,
        limit: Optional[int] = None,
    ) -> List[TradeEvent]:
        filtered = [
            event
            for event in self.events
            if {event.proposer, event.target} == {player_a, player_b}
        ]

        if limit is None or limit >= len(filtered):
            return filtered
        return filtered[-limit:]

    def get_acceptance_rate(self, proposer: Optional[PlayerId] = None, target: Optional[PlayerId] = None) -> float:
        relevant = self.events

        if proposer is not None:
            relevant = [event for event in relevant if event.proposer == proposer]

        if target is not None:
            relevant = [event for event in relevant if event.target == target]

        if len(relevant) == 0:
            return 0.0

        accepted = sum(1 for event in relevant if event.accepted)
        return accepted / len(relevant)

    def get_counter_rate(self, proposer: Optional[PlayerId] = None, target: Optional[PlayerId] = None) -> float:
        relevant = self.events

        if proposer is not None:
            relevant = [event for event in relevant if event.proposer == proposer]

        if target is not None:
            relevant = [event for event in relevant if event.target == target]

        if len(relevant) == 0:
            return 0.0

        counters = sum(1 for event in relevant if event.response_type == "counter")
        return counters / len(relevant)

    def get_trade_volume(self, player_id: Optional[PlayerId] = None) -> int:
        relevant = self.events
        if player_id is not None:
            relevant = [
                event
                for event in relevant
                if event.proposer == player_id or event.target == player_id
            ]

        total = 0
        for event in relevant:
            total += sum(event.offer.values())
            total += sum(event.request.values())
        return total

    def build_sequence_tensor_dict(self, limit: int = 16) -> dict:
        """
        Returns a raw sequence representation for later encoding.

        This stays Python-native so the environment can expose it cleanly.
        Tensor conversion should happen in the model input pipeline.
        """
        recent = self.get_recent_events(limit=limit)

        proposer_ids = []
        target_ids = []
        response_types = []
        offers = []
        requests = []
        counter_offers = []
        counter_requests = []
        accepted_flags = []
        executed_flags = []
        turn_numbers = []
        vp_deltas = []

        response_map = {
            "reject": 0,
            "accept": 1,
            "counter": 2,
            "noop": 3,
        }

        for event in recent:
            proposer_ids.append(int(event.proposer))
            target_ids.append(int(event.target))
            response_types.append(response_map.get(event.response_type, 3))
            offers.append([event.offer[r] for r in empty_trade_vector().keys()])
            requests.append([event.request[r] for r in empty_trade_vector().keys()])
            counter_offers.append([event.counter_offer[r] for r in empty_trade_vector().keys()])
            counter_requests.append([event.counter_request[r] for r in empty_trade_vector().keys()])
            accepted_flags.append(1 if event.accepted else 0)
            executed_flags.append(1 if event.executed else 0)
            turn_numbers.append(event.turn_number)
            vp_deltas.append(
                [
                    event.proposer_vp_after - event.proposer_vp_before,
                    event.target_vp_after - event.target_vp_before,
                ]
            )

        return {
            "proposer_ids": proposer_ids,
            "target_ids": target_ids,
            "response_types": response_types,
            "offers": offers,
            "requests": requests,
            "counter_offers": counter_offers,
            "counter_requests": counter_requests,
            "accepted_flags": accepted_flags,
            "executed_flags": executed_flags,
            "turn_numbers": turn_numbers,
            "vp_deltas": vp_deltas,
        }

    def summary_for_player(self, player_id: PlayerId) -> dict:
        relevant = self.get_recent_events_for_player(player_id)

        if len(relevant) == 0:
            return {
                "num_trades": 0,
                "acceptance_rate": 0.0,
                "counter_rate": 0.0,
                "trade_volume": 0,
            }

        accepts = sum(1 for event in relevant if event.accepted)
        counters = sum(1 for event in relevant if event.response_type == "counter")
        volume = 0

        for event in relevant:
            volume += sum(event.offer.values())
            volume += sum(event.request.values())

        return {
            "num_trades": len(relevant),
            "acceptance_rate": accepts / len(relevant),
            "counter_rate": counters / len(relevant),
            "trade_volume": volume,
        }