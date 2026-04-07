import pytest

from environment.catan_env import CatanEnv
from core.constants import PlayerId, Resource, BuildingType
from core.phase_router import TurnPhase
from core.constructions import Building


def test_7_forces_discard_and_robber_move_and_steal():
    env = CatanEnv(seed=123, enable_trading=False)
    obs = env.reset()

    engine = env.engine
    # Place resources to force discard for WHITE, no discard for BLUE
    engine.players[PlayerId.WHITE].resources = {
        Resource.WOOD: 3,
        Resource.BRICK: 3,
        Resource.SHEEP: 3,
        Resource.WHEAT: 0,
        Resource.ORE: 0,
    }
    engine.players[PlayerId.BLUE].resources = {
        Resource.WOOD: 1,
        Resource.BRICK: 0,
        Resource.SHEEP: 0,
        Resource.WHEAT: 0,
        Resource.ORE: 0,
    }

    # Find a tile that does not currently have the robber and place a Blue settlement on it
    target_tile = next(t for t in engine.board.tiles if not t.has_robber)
    vertex = target_tile.vertices[0]
    vertex.place_building(Building(BuildingType.SETTLEMENT, PlayerId.BLUE, vertex))
    engine.players[PlayerId.BLUE].n_settlements = 1

    # Simulate the robber trigger (7 roll)
    engine.robber_pending = True
    engine._apply_robber_effects()
    engine.phase_router.set_phase(TurnPhase.MAIN_ACTION)

    assert engine.last_robber_event["rolled_seven"] is True
    assert engine.last_robber_event["robber_moved"] is False

    # White must discard 4 as action now.
    assert engine.robber_discard_required[PlayerId.WHITE] == 4
    assert engine.robber_discard_queue == [PlayerId.WHITE]

    discard_reward = engine.apply_gameplay_action({"type": "discard_cards", "resources": {Resource.WOOD: 2, Resource.BRICK: 2}})
    assert discard_reward == 0
    white_total = sum(engine.players[PlayerId.WHITE].resources.values())
    assert white_total == 5

    assert engine.robber_discard_queue == []

    # Move robber to the selected target tile and steal from BLUE (only adjacent with resources)
    engine._move_robber(target_tile.id)

    assert engine.board.tiles[target_tile.id].has_robber is True
    assert engine.robber_pending is False
    assert engine.last_robber_event["robber_moved"] is True
    assert engine.last_robber_event["moved_to"] == str(target_tile.id)

    assert engine.players[PlayerId.BLUE].resources[Resource.WOOD] == 0
    assert engine.players[PlayerId.WHITE].resources[Resource.WOOD] == 2


def test_robber_pending_blocks_non_move_actions():
    env = CatanEnv(seed=999, enable_trading=False)
    env.reset()

    engine = env.engine
    engine.robber_pending = True
    engine.last_robber_event = {"rolled_seven": True, "discarded": {}, "robber_moved": False}

    # Attempt a build action while robber is pending should be rejected and robber still pending.
    reward = engine.apply_gameplay_action({"type": "build_road", "connection": 0})
    assert reward < 0
    assert engine.robber_pending is True

    # Moving robber clears pending.
    # Choose an available non-robber tile so this works deterministically.
    target_tile = next(t for t in engine.board.tiles if not t.has_robber)
    engine.players[PlayerId.WHITE].n_settlements = 1
    # place an adjacent settlement to have a victim for steal attempt
    vertex = target_tile.vertices[0]
    vertex.place_building(Building(BuildingType.SETTLEMENT, PlayerId.BLUE, vertex))
    engine.players[PlayerId.BLUE].n_settlements = 1

    engine.phase_router.set_phase(TurnPhase.MAIN_ACTION)
    reward = engine.apply_gameplay_action({"type": "move_robber", "tile": target_tile.id})
    assert reward > 0
    assert engine.robber_pending is False


def test_7_discard_as_action_then_robber_move():
    env = CatanEnv(seed=111, enable_trading=False)
    env.reset()

    engine = env.engine
    engine.players[PlayerId.WHITE].resources = {
        Resource.WOOD: 3,
        Resource.BRICK: 3,
        Resource.SHEEP: 3,
        Resource.WHEAT: 1,
        Resource.ORE: 0,
    }
    engine.players[PlayerId.BLUE].resources = {
        Resource.WOOD: 1,
        Resource.BRICK: 0,
        Resource.SHEEP: 0,
        Resource.WHEAT: 0,
        Resource.ORE: 0,
    }

    engine.robber_pending = True
    engine.last_robber_event = {"rolled_seven": True, "discarded": {}, "robber_moved": False}
    engine._apply_robber_effects()
    engine.phase_router.set_phase(TurnPhase.MAIN_ACTION)

    assert engine.robber_discard_required[PlayerId.WHITE] == 5
    assert engine.robber_discard_queue == [PlayerId.WHITE]

    # invalid resource selection: too few
    bad_reward = engine.apply_gameplay_action({"type": "discard_cards", "resources": {Resource.WOOD: 2, Resource.BRICK: 1}})
    assert bad_reward < 0

    # valid discard
    good_reward = engine.apply_gameplay_action({"type": "discard_cards", "resources": {Resource.WOOD: 2, Resource.BRICK: 2, Resource.SHEEP: 1}})
    assert good_reward == 0
    assert PlayerId.WHITE not in engine.robber_discard_required
    assert engine.robber_discard_queue == []

    target_tile = next(t for t in engine.board.tiles if not t.has_robber)
    vertex = target_tile.vertices[0]
    vertex.place_building(Building(BuildingType.SETTLEMENT, PlayerId.BLUE, vertex))
    engine.players[PlayerId.BLUE].n_settlements = 1

    move_reward = engine.apply_gameplay_action({"type": "move_robber", "tile": target_tile.id})
    assert move_reward > 0
    assert engine.robber_pending is False
