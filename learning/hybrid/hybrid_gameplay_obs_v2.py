from __future__ import annotations

from typing import List

from core.constants import Resource
from environment.catan_env import CatanEnv


def _resource_count(resources: dict, resource: Resource) -> float:
    return float(resources.get(resource.name, resources.get(resource, 0)))


def build_gameplay_state_vector(env: CatanEnv) -> List[float]:
    raw = env.get_observation()
    game = raw.get("game", {})
    player = raw.get("player", {})
    players = raw.get("players", {})
    current_player = game.get("current_player")

    resources = player.get("resources", {})
    others = [state for pid, state in players.items() if pid != current_player]

    vec = [
        _resource_count(resources, Resource.WOOD),
        _resource_count(resources, Resource.BRICK),
        _resource_count(resources, Resource.SHEEP),
        _resource_count(resources, Resource.WHEAT),
        _resource_count(resources, Resource.ORE),
        float(player.get("victory_points", 0)),
        float(player.get("num_settlements", 0)),
        float(player.get("num_cities", 0)),
        float(player.get("roads", 0) if not isinstance(player.get("roads", 0), list) else len(player.get("roads", []))),
        float(player.get("bonus_vp", 0)),
        float(player.get("dev_victory_points", 0)),
        float(player.get("played_knights", 0)),
        float(player.get("revealed_vp_cards", 0)),
        float(game.get("turn_number", 0)),
        float(env.get_phase().value),
        float(game.get("last_roll") or 0),
        1.0 if game.get("robber_pending", False) else 0.0,
        float(game.get("dev_card_deck_size", 0)),
        1.0 if game.get("longest_road_owner") == current_player else 0.0,
        1.0 if game.get("largest_army_owner") == current_player else 0.0,
        float(len(raw.get("legal_actions", []))),
        float(len(env.engine.robber_discard_queue)),
        float(env.engine.robber_discard_required.get(env.get_current_player_id(), 0)),
    ]

    for state in others[:3]:
        other_resources = state.get("resources", {})
        vec.extend(
            [
                _resource_count(other_resources, Resource.WOOD),
                _resource_count(other_resources, Resource.BRICK),
                _resource_count(other_resources, Resource.SHEEP),
                _resource_count(other_resources, Resource.WHEAT),
                _resource_count(other_resources, Resource.ORE),
                float(state.get("victory_points", 0)),
                float(state.get("num_settlements", 0)),
                float(state.get("num_cities", 0)),
                float(state.get("roads", 0) if not isinstance(state.get("roads", 0), list) else len(state.get("roads", []))),
                float(state.get("bonus_vp", 0)),
                float(state.get("played_knights", 0)),
                float(state.get("revealed_vp_cards", 0)),
                0.0,
            ]
        )

    vec += [0.0] * max(0, 64 - len(vec))
    return vec[:64]
