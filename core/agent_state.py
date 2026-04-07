from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.constants import (
    COLLECTABLE_RESOURCES,
    DevCard,
    PlayerId,
    Resource,
    ResourceCost,
    VP_CITY,
    VP_SETTLEMENT,
    cost_to_dict,
    empty_hand,
)


def _add_hand_inplace(hand: Dict[Resource, int], delta: Dict[Resource, int]) -> None:
    for resource, value in delta.items():
        if resource == Resource.DESERT:
            continue
        hand[resource] = hand.get(resource, 0) + int(value)


def _sub_hand_inplace(hand: Dict[Resource, int], delta: Dict[Resource, int]) -> None:
    for resource, value in delta.items():
        if resource == Resource.DESERT:
            continue
        hand[resource] = hand.get(resource, 0) - int(value)


@dataclass
class AgentState:
    player_id: PlayerId

    resources: Dict[Resource, int] = field(default_factory=empty_hand)
    dev_cards: List[DevCard] = field(default_factory=list)
    new_dev_cards: List[DevCard] = field(default_factory=list)
    played_dev_cards_sequence: List[DevCard] = field(default_factory=list)

    played_dev_card_this_turn: bool = False

    buildings: List[object] = field(default_factory=list)
    roads: List[int] = field(default_factory=list)

    hidden_vp_cards: int = 0
    revealed_vp_cards: int = 0
    played_knights: int = 0

    n_settlements: int = 0
    n_cities: int = 0
    n_roads: int = 0

    bonus_vp: int = 0
    longest_road_length: int = 0
    dev_victory_points: int = 0
    victory_points: int = 0

    def reset_turn_flags(self) -> None:
        if self.new_dev_cards:
            self.dev_cards.extend(self.new_dev_cards)
            self.new_dev_cards = []
        self.played_dev_card_this_turn = False

    def reset_for_new_game(self) -> None:
        self.resources = empty_hand()
        self.dev_cards = []
        self.new_dev_cards = []
        self.played_dev_cards_sequence = []
        self.played_dev_card_this_turn = False
        self.buildings = []
        self.roads = []
        self.hidden_vp_cards = 0
        self.revealed_vp_cards = 0
        self.played_knights = 0
        self.n_settlements = 0
        self.n_cities = 0
        self.n_roads = 0
        self.bonus_vp = 0
        self.longest_road_length = 0
        self.dev_victory_points = 0
        self.victory_points = 0

    def add_resource(self, resource: Resource, amount: int = 1) -> None:
        if resource == Resource.DESERT:
            return
        self.resources[resource] = self.resources.get(resource, 0) + int(amount)

    def remove_resource(self, resource: Resource, amount: int = 1) -> None:
        if resource == Resource.DESERT:
            return
        self.resources[resource] = self.resources.get(resource, 0) - int(amount)

    def receive(self, delta: Dict[Resource, int]) -> None:
        _add_hand_inplace(self.resources, delta)

    def pay_resources(self, delta: Dict[Resource, int]) -> None:
        _sub_hand_inplace(self.resources, delta)

    def can_pay_cost(self, cost: ResourceCost) -> bool:
        cost_dict = cost_to_dict(cost)
        for resource, needed in cost_dict.items():
            if needed <= 0:
                continue
            if self.resources.get(resource, 0) < needed:
                return False
        return True

    def pay_cost(self, cost: ResourceCost) -> None:
        cost_dict = cost_to_dict(cost)
        _sub_hand_inplace(self.resources, cost_dict)

    def total_resources(self) -> int:
        return sum(self.resources.get(resource, 0) for resource in COLLECTABLE_RESOURCES)

    def discard_half_if_needed(self) -> int:
        total = self.total_resources()
        if total <= 7:
            return 0
        return total // 2

    def add_dev_card(self, card: DevCard, playable: bool = False) -> None:
        if playable:
            self.dev_cards.append(card)
        else:
            self.new_dev_cards.append(card)

    def count_dev_card(self, card: DevCard) -> int:
        return sum(1 for c in self.dev_cards if c == card)

    def can_play_dev_card(self, card: DevCard) -> bool:
        if self.played_dev_card_this_turn:
            return False
        return self.count_dev_card(card) > 0

    def play_dev_card(self, card: DevCard) -> None:
        if not self.can_play_dev_card(card):
            raise ValueError(f"Cannot play dev card {card} right now.")

        for i, owned_card in enumerate(self.dev_cards):
            if owned_card == card:
                self.dev_cards.pop(i)
                self.played_dev_card_this_turn = True
                self.played_dev_cards_sequence.append(card)
                return

        raise ValueError(f"Dev card {card} not found in hand.")

    def num_settlements(self) -> int:
        return int(self.n_settlements)

    def num_cities(self) -> int:
        return int(self.n_cities)

    def num_roads(self) -> int:
        return int(self.n_roads)

    def piece_vp(self) -> int:
        return self.num_settlements() * VP_SETTLEMENT + self.num_cities() * VP_CITY

    def total_dev_vp(self) -> int:
        return int(self.hidden_vp_cards) + int(self.revealed_vp_cards)

    def public_vp(self) -> int:
        return self.piece_vp() + int(self.bonus_vp) + int(self.revealed_vp_cards)

    def total_vp(self) -> int:
        return self.piece_vp() + int(self.bonus_vp) + self.total_dev_vp()

    def update_victory_points(self) -> int:
        self.victory_points = self.total_vp()
        return self.victory_points

    def as_dict(self, private: bool = False) -> dict:
        self.update_victory_points()
        dev_cards_value: list[str] | int
        if private:
            dev_cards_value = [card.name for card in (self.dev_cards + self.new_dev_cards)]
        else:
            dev_cards_value = len(self.dev_cards) + len(self.new_dev_cards) + int(self.hidden_vp_cards)

        return {
            "player_id": self.player_id.name,
            "resources": {resource.name: self.resources.get(resource, 0) for resource in COLLECTABLE_RESOURCES},
            "dev_cards": dev_cards_value,
            "played_dev_card_this_turn": self.played_dev_card_this_turn,
            "played_dev_cards_sequence": [c.name for c in self.played_dev_cards_sequence],
            "roads": list(self.roads),
            "num_buildings": self.num_settlements() + self.num_cities(),
            "num_settlements": self.num_settlements(),
            "num_cities": self.num_cities(),
            "num_roads": self.num_roads(),
            "longest_road_length": int(self.longest_road_length),
            "hidden_vp_cards": int(self.hidden_vp_cards) if private else 0,
            "played_knights": int(self.played_knights),
            "bonus_vp": int(self.bonus_vp),
            "revealed_vp_cards": int(self.revealed_vp_cards),
            "dev_victory_points": int(self.total_dev_vp()) if private else int(self.revealed_vp_cards),
            "victory_points": int(self.victory_points) if private else int(self.public_vp()),
        }
