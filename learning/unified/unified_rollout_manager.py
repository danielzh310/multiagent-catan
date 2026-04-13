from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.constants import PlayerId, Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv

def _worker_build_obs(
    obs_raw: Dict,
    legal_actions: List[Dict],
    player_stats: Dict,
    pending_info: Optional[Dict],
    phase: TurnPhase,
) -> Dict[str, np.ndarray]:
    """
    Builds the obs numpy array dict from pre-fetched state data.
    This runs in the worker process to offload CPU work from the main process.
    """
    player = obs_raw["player"]
    other_players = [
        v for k, v in obs_raw["players"].items()
        if k != obs_raw["game"]["current_player"]
    ]

    def to_vec(state: dict) -> np.ndarray:
        resources = state.get("resources", {})
        roads_val = state.get("roads", 0)
        if isinstance(roads_val, list):
            roads_val = len(roads_val)
        dev_cards_val = state.get("dev_cards", 0)
        if isinstance(dev_cards_val, list):
            dev_cards_val = len(dev_cards_val)
        vec = np.array([
            float(resources.get("WOOD", 0)),
            float(resources.get("BRICK", 0)),
            float(resources.get("SHEEP", 0)),
            float(resources.get("WHEAT", 0)),
            float(resources.get("ORE", 0)),
            float(state.get("victory_points", 0)),
            float(state.get("num_settlements", 0)),
            float(state.get("num_cities", 0)),
            float(roads_val),
            float(state.get("bonus_vp", 0)),
            float(state.get("dev_victory_points", 0)),
            float(dev_cards_val),
            float(state.get("played_knights", 0)),
            float(state.get("revealed_vp_cards", 0)),
        ], dtype=np.float32)
        padded = np.zeros(64, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded

    self_np = to_vec(player)

    if other_players:
        op_np = np.mean([to_vec(opp) for opp in other_players], axis=0).astype(np.float32)
    else:
        op_np = np.zeros(64, dtype=np.float32)

    board_np = np.zeros(64, dtype=np.float32)
    game = obs_raw.get("game", {})
    board_np[0] = float(game.get("turn_number", 0))
    board_np[1] = float(int(obs_raw["game"].get("current_player", 0)))
    board_np[2] = float(phase.value)
    board_np[3] = 1.0 if game.get("enable_trading", True) else 0.0

    last_roll = game.get("last_roll")
    board_np[4] = float(last_roll if last_roll is not None else 0.0)
    board_np[5] = 1.0 if game.get("robber_pending", False) else 0.0

    if pending_info is not None:
        board_np[6] = 1.0
        board_np[7] = pending_info["counter_count"]

    robber_event = game.get("last_robber_event")
    if robber_event is not None:
        board_np[8] = 1.0 if robber_event.get("rolled_seven", False) else 0.0
        board_np[9] = 1.0 if robber_event.get("stolen_from") is not None else 0.0
        discarded = robber_event.get("discarded", {})
        total_discarded = sum(float(sum(res_map.values())) for res_map in discarded.values())
        board_np[10] = total_discarded

    board_np[11] = float(len(legal_actions))
    board_np[12] = 1.0 if game.get("initial_placement_phase", False) else 0.0
    board_np[13] = float(game.get("initial_placement_index", 0))
    stage = game.get("initial_placement_stage")
    board_np[14] = 1.0 if stage == "settlement" else 0.0
    board_np[15] = 1.0 if stage == "road" else 0.0
    board_np[16] = float(game.get("dev_card_deck_size", 0))

    current_player_id = obs_raw["game"].get("current_player")
    board_np[17] = 1.0 if game.get("longest_road_owner") == current_player_id else 0.0
    board_np[18] = 1.0 if game.get("largest_army_owner") == current_player_id else 0.0

    board_np[19] = float(len(obs_raw.get("robber_discard_queue", [])))
    board_np[20] = float(obs_raw.get("robber_discard_required", {}).get(current_player_id, 0))

    gameplay_np, gameplay_mask_np = _encode_gameplay_actions_batch(
        legal_actions, player_stats,
        UnifiedRolloutManager.MAX_GAMEPLAY_ACTIONS, UnifiedRolloutManager.GAMEPLAY_FEATURE_DIM,
    )
    trade_np, trade_mask_np = _encode_trade_actions_batch(
        legal_actions, player_stats, pending_info,
        UnifiedRolloutManager.MAX_TRADE_ACTIONS, UnifiedRolloutManager.TRADE_FEATURE_DIM,
    )

    return {
        "board": board_np, "self": self_np, "opponent": op_np,
        "gameplay_candidates": gameplay_np, "gameplay_mask": gameplay_mask_np,
        "trade_candidates": trade_np, "trade_mask": trade_mask_np,
    }

def _worker(
    worker_id: int,
    env_indices: List[int],
    cmd_pipe: Connection,
    enable_trading: bool,
    max_steps: Optional[int],
) -> None:
    """
    Runs a subset of CatanEnv instances in a separate process.
    Listens on `cmd_pipe` for commands and sends results back.

    Protocol (all messages are plain Python objects, pickle-able):
      cmd: ("reset", [idx, ...])          -> list of (idx, obs_dict)
      cmd: ("step", [(idx, action), ...]) -> list of (idx, obs_dict, reward, done, info)
      cmd: ("get_obs", [idx, ...])        -> list of (idx, obs_dict)
      cmd: ("close",)                     -> exits loop
    """
    # Build only the envs owned by this worker
    envs: Dict[int, CatanEnv] = {}
    for idx in env_indices:
        envs[idx] = CatanEnv(enable_trading=enable_trading, max_steps=max_steps)
        envs[idx].reset()

    try:
        while True:
            cmd, payload = cmd_pipe.recv()

            if cmd == "reset":
                results = []
                for idx in payload:
                    obs = envs[idx].reset()
                    results.append((idx, obs))
                cmd_pipe.send(results)

            elif cmd == "step":
                results = []
                for idx, action in payload:
                    _, reward, done, info = envs[idx].step(action)
                    if done:
                        obs = envs[idx].reset()
                    else:
                        obs = envs[idx].get_observation()
                    results.append((idx, obs, reward, done, info))
                cmd_pipe.send(results)

            elif cmd == "get_phase":
                results = []
                for idx in payload:
                    phase = envs[idx].get_phase()
                    results.append((idx, phase))
                cmd_pipe.send(results)

            elif cmd == "get_legal_actions":
                results = []
                for idx in payload:
                    legal = envs[idx].get_legal_actions()
                    current_player = envs[idx].get_current_player_id()
                    player = envs[idx].engine.players[current_player]
                    player_stats = {
                        "vp": float(player.update_victory_points()),
                        "n_settlements": float(player.n_settlements),
                        "n_cities": float(player.n_cities),
                        "n_roads": float(player.n_roads),
                        "resources": {str(k): float(v) for k, v in player.resources.items()},
                    }
                    pending = envs[idx].engine.trade_manager.get_pending_trade()
                    pending_info = None
                    if pending is not None:
                        pending_info = {
                            "counter_count": float(pending.counter_count),
                            "proposer": float(int(pending.proposer)),
                            "target": float(int(pending.target)),
                        }
                    results.append((idx, legal, envs[idx].get_observation(), player_stats, pending_info))
                cmd_pipe.send(results)

            elif cmd == "get_full_state":
                """Returns everything needed to build obs tensors in the main process."""
                results = []
                for idx in payload:
                    env = envs[idx]
                    phase = env.get_phase()
                    legal = env.get_legal_actions()
                    current_player = env.get_current_player_id()
                    player = env.engine.players[current_player]

                    player_stats = {
                        "vp": float(player.update_victory_points()),
                        "n_settlements": float(player.n_settlements),
                        "n_cities": float(player.n_cities),
                        "n_roads": float(player.n_roads),
                        "resources": {str(k): float(v) for k, v in player.resources.items()},
                        "resource_total": float(sum(int(v) for v in player.resources.values())),
                    }

                    pending = env.engine.trade_manager.get_pending_trade()
                    pending_info = None
                    if pending is not None:
                        pending_info = {
                            "counter_count": float(pending.counter_count),
                            "proposer": float(int(pending.proposer)),
                            "target": float(int(pending.target)),
                        }

                    obs_np_dict = _worker_build_obs(
                        env.get_observation(), legal, player_stats, pending_info, phase
                    )

                    results.append((idx, phase, legal, obs_np_dict, player_stats, pending_info))
                cmd_pipe.send(results)

            elif cmd == "get_game_stats":
                results = []
                for idx in payload:
                    env = envs[idx]
                    results.append((idx, {
                        "winner": env.engine.winner,
                        "total_steps": env.get_step_count(),
                        "completed_naturally": (env.engine.winner is not None),
                    }))
                cmd_pipe.send(results)

            elif cmd == "close":
                break

    except Exception:
        traceback.print_exc()
    finally:
        cmd_pipe.close()
class AsyncVectorEnv:
    """
    Spawns `num_workers` child processes, each owning `envs_per_worker` CatanEnvs.
    The main process sends batched commands and receives results asynchronously,
    keeping all CPU cores busy while the GPU processes the previous batch.
    """

    def __init__(
        self,
        num_envs: int,
        num_workers: int,
        enable_trading: bool = True,
        max_steps: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.num_workers = min(num_workers, num_envs)

        # Partition env indices across workers as evenly as possible
        indices = list(range(num_envs))
        self._worker_env_indices: List[List[int]] = [[] for _ in range(self.num_workers)]
        for i, idx in enumerate(indices):
            self._worker_env_indices[i % self.num_workers].append(idx)

        # Build index -> worker mapping for fast dispatch
        self._env_to_worker: Dict[int, int] = {}
        for w, env_list in enumerate(self._worker_env_indices):
            for idx in env_list:
                self._env_to_worker[idx] = w

        ctx = mp.get_context("spawn")
        process_factory = getattr(ctx, "Process")
        self._pipes: List[Connection] = []
        self._procs: List[mp.Process] = []

        for w in range(self.num_workers):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = process_factory(
                target=_worker,
                args=(w, self._worker_env_indices[w], child_conn, enable_trading, max_steps),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # close child end in parent
            self._pipes.append(parent_conn) # type: ignore
            self._procs.append(proc)

    def _dispatch(self, cmd: str, payloads: Dict[int, List]) -> Dict[int, Any]:
        """
        Sends `cmd` with per-worker payloads to all workers that have work,
        then collects and merges results keyed by env index.
        """
        active_workers = []
        for w in range(self.num_workers):
            worker_payload = payloads.get(w, [])
            if worker_payload:
                self._pipes[w].send((cmd, worker_payload))
                active_workers.append(w)

        results: Dict[int, Any] = {}
        for w in active_workers:
            worker_results = self._pipes[w].recv()
            for item in worker_results:
                idx = item[0]
                results[idx] = item[1:]  # strip the leading env index

        return results

    def get_full_state(self, env_indices: Optional[List[int]] = None) -> Dict[int, tuple]:
        """
        Returns (phase, legal_actions, obs_raw, player_stats, pending_info) per env.
        """
        if env_indices is None:
            env_indices = list(range(self.num_envs))

        payloads: Dict[int, List[int]] = {}
        for idx in env_indices:
            w = self._env_to_worker[idx]
            payloads.setdefault(w, []).append(idx)

        return self._dispatch("get_full_state", payloads)

    def step(self, actions: Dict[int, Any]) -> Dict[int, tuple]:
        """
        actions: {env_idx: action_dict}
        Returns {env_idx: (obs_raw, reward, done, info)}
        """
        payloads: Dict[int, List] = {}
        for idx, action in actions.items():
            w = self._env_to_worker[idx]
            payloads.setdefault(w, []).append((idx, action))

        return self._dispatch("step", payloads)

    def reset(self, env_indices: Optional[List[int]] = None) -> Dict[int, Any]:
        if env_indices is None:
            env_indices = list(range(self.num_envs))

        payloads: Dict[int, List[int]] = {}
        for idx in env_indices:
            w = self._env_to_worker[idx]
            payloads.setdefault(w, []).append(idx)

        return self._dispatch("reset", payloads)

    def get_game_stats(self, env_indices: List[int]) -> Dict[int, Dict]:
        payloads: Dict[int, List[int]] = {}
        for idx in env_indices:
            w = self._env_to_worker[idx]
            payloads.setdefault(w, []).append(idx)
        raw = self._dispatch("get_game_stats", payloads)
        return {idx: v[0] for idx, v in raw.items()}

    def close(self):
        for w in range(self.num_workers):
            try:
                self._pipes[w].send(("close", None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()

_ACTION_TYPE_TO_IDX: Dict[str, int] = {
    "build_settlement": 0, "build_road": 1, "build_city": 2,
    "buy_dev_card": 3, "play_dev_card": 4, "bank_trade": 5,
    "move_robber": 6, "discard_cards": 7, "end_main_action": 8,
    "end_turn": 9, "roll": 10, "skip_trade": 11,
}

_TRADE_ACTION_TYPE_TO_IDX: Dict[str, int] = {
    "skip_trade": 0, "propose_trade": 1, "accept_trade": 2,
    "reject_trade": 3, "counter_trade": 4,
}

_RESOURCE_NAMES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
_RESOURCE_TO_SLOT = {r: i for i, r in enumerate(_RESOURCE_NAMES)}

def _res_slot(r: Any) -> int:
    if r is None:
        return -1
    try:
        name = r.name if hasattr(r, "name") else str(r)
        return _RESOURCE_TO_SLOT.get(name, -1)
    except Exception:
        return -1


def _encode_gameplay_actions_batch(
    actions: List[Dict[str, Any]],
    player_stats: Dict[str, Any],
    max_actions: int,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized encoding: builds (max_actions, feature_dim) float32 array
    and a boolean mask in one NumPy pass. Replaces the per-action Python loop.
    """
    n = min(len(actions), max_actions)
    out = np.zeros((max_actions, feature_dim), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=bool)

    if n == 0:
        mask[0] = True
        return out, mask

    vp = player_stats["vp"]
    n_settlements = player_stats["n_settlements"]
    n_cities = player_stats["n_cities"]
    n_roads = player_stats["n_roads"]

    for i in range(n):
        a = actions[i]
        f = out[i]

        # One-hot action type (features 0-11)
        atype = a.get("type", "")
        tidx = _ACTION_TYPE_TO_IDX.get(atype)
        if tidx is not None:
            f[tidx] = 1.0

        # Player state features (12-15)
        f[12] = vp / 10.0
        f[13] = n_settlements / 5.0
        f[14] = n_cities / 4.0
        f[15] = n_roads / 15.0

        # Spatial features (16-20)
        v = a.get("vertex")
        if v is not None:
            f[16] = float(v) / 64.0
        c = a.get("connection")
        if c is not None:
            f[17] = float(c) / 128.0
        t = a.get("tile")
        if t is not None:
            f[18] = float(t) / 19.0
        c1 = a.get("connection_1")
        if c1 is not None:
            f[19] = float(c1) / 128.0
        c2 = a.get("connection_2")
        if c2 is not None:
            f[20] = float(c2) / 128.0

        # Resource features (21-38)
        give_slot = _res_slot(a.get("give"))
        recv_slot = _res_slot(a.get("receive"))
        res_slot  = _res_slot(a.get("resource"))
        res1_slot = _res_slot(a.get("resource_1"))
        res2_slot = _res_slot(a.get("resource_2"))

        if give_slot >= 0:
            f[21 + give_slot] = 1.0
        if recv_slot >= 0:
            f[26 + recv_slot] = 1.0
        if res_slot >= 0:
            f[31] = float(res_slot + 1) / 5.0
        if res1_slot >= 0:
            f[32] = float(res1_slot + 1) / 5.0
        if res2_slot >= 0:
            f[33] = float(res2_slot + 1) / 5.0

        card = a.get("card")
        if card is not None:
            try:
                f[34] = float(int(card) + 1) / 5.0
            except (ValueError, TypeError):
                pass

        rate = a.get("rate")
        if rate is not None:
            f[35] = float(rate) / 4.0
        req = a.get("required")
        if req is not None:
            f[36] = float(req) / 8.0

        discard = a.get("resources")
        if isinstance(discard, dict):
            vals = list(discard.values())
            total = float(sum(int(v) for v in vals))
            non_zero = float(sum(1 for v in vals if int(v) > 0))
            f[37] = total / 8.0
            f[38] = non_zero / 5.0

        if atype == "play_dev_card":
            f[39] = 1.0

        mask[i] = True

    return out, mask


def _encode_trade_actions_batch(
    actions: List[Dict[str, Any]],
    player_stats: Dict[str, Any],
    pending_info: Optional[Dict],
    max_actions: int,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized trade action encoding. Replaces the per-action Python loop.
    """
    n = min(len(actions), max_actions)
    out = np.zeros((max_actions, feature_dim), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=bool)

    if n == 0:
        mask[0] = True
        return out, mask

    vp = player_stats["vp"]
    res_total = player_stats["resource_total"]

    p_counter = float(pending_info["counter_count"]) if pending_info else 0.0
    p_proposer = float(pending_info["proposer"]) if pending_info else 0.0
    p_target = float(pending_info["target"]) if pending_info else 0.0
    has_pending = 1.0 if pending_info is not None else 0.0

    for i in range(n):
        a = actions[i]
        f = out[i]

        # One-hot action type (0-4)
        atype = a.get("type", "")
        tidx = _TRADE_ACTION_TYPE_TO_IDX.get(atype)
        if tidx is not None:
            f[tidx] = 1.0

        # Player state (5-6)
        f[5] = vp / 10.0
        f[6] = res_total / 20.0

        # Pending trade context (7-10)
        f[7] = has_pending
        if has_pending:
            f[8] = p_counter / 3.0
            f[9] = p_proposer / 3.0
            f[10] = p_target / 3.0

        # Target player one-hot (11-14)
        target = a.get("target")
        if target is not None:
            try:
                f[11 + int(target)] = 1.0
            except (ValueError, TypeError, IndexError):
                pass

        # Offer/request resource amounts (15-24)
        offer = a.get("offer") or a.get("counter_offer") or {}
        request = a.get("request") or a.get("counter_request") or {}

        offer_total = 0.0
        for resource, amount in offer.items():
            slot = _res_slot(resource)
            if slot >= 0:
                v = float(amount)
                f[15 + slot] = v
                offer_total += v

        request_total = 0.0
        for resource, amount in request.items():
            slot = _res_slot(resource)
            if slot >= 0:
                v = float(amount)
                f[20 + slot] = v
                request_total += v

        f[25] = offer_total / 4.0
        f[26] = request_total / 4.0

        # Response type (27-29)
        response_type = a.get("response_type", "")
        if response_type == "accept":
            f[27] = 1.0
        elif response_type == "reject":
            f[28] = 1.0
        elif response_type == "counter":
            f[29] = 1.0

        if atype == "counter_trade":
            f[30] = 1.0
        if atype == "propose_trade":
            f[31] = 1.0

        mask[i] = True

    return out, mask
class UnifiedRolloutManager:
    MAX_GAMEPLAY_ACTIONS = 256
    GAMEPLAY_FEATURE_DIM = 40
    MAX_TRADE_ACTIONS = 128
    TRADE_FEATURE_DIM = 32

    def __init__(
        self,
        num_envs: int,
        device: str = "cpu",
        enable_trading: bool = True,
        max_steps: int | None = None,
        num_workers: int | None = None,
    ):
        self.num_envs = num_envs
        self.device = device
        self.enable_trading = enable_trading
        self.game_stats_buffer: List[Dict] = []

        # Default to number of available CPUs, capped to num_envs
        if num_workers is None:
            num_workers = min(mp.cpu_count(), num_envs)
        self._num_workers = num_workers

        # Async vector env: all Catan logic runs in parallel worker processes
        self._vec_env = AsyncVectorEnv(
            num_envs=num_envs,
            num_workers=num_workers,
            enable_trading=enable_trading,
            max_steps=max_steps,
        )

    def _phase_name(self, phase: TurnPhase) -> str:
        if phase in (TurnPhase.SETUP, TurnPhase.MAIN_ACTION, TurnPhase.END_TURN):
            return "gameplay"
        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            return "trade"
        return "auto"

    def _resource_slot(self, resource_value: Any) -> int:
        return _res_slot(resource_value)

    def _resolve_discard_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        required = int(action.get("required", 0))
        available = action.get("available", {})
        ordered_resources = sorted(available.items(), key=lambda item: (-int(item[1]), item[0]))
        resources_to_discard: Dict[Resource, int] = {
            Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 0,
            Resource.WHEAT: 0, Resource.ORE: 0,
        }
        remaining = required
        for resource_name, count in ordered_resources:
            if remaining <= 0:
                break
            take = min(int(count), remaining)
            if take <= 0:
                continue
            try:
                resource = Resource[resource_name]
            except KeyError:
                continue
            resources_to_discard[resource] = take
            remaining -= take
        if remaining > 0:
            return action
        resolved = dict(action)
        resolved["resources"] = resources_to_discard
        return resolved

    def _decode_gameplay(self, action_idx: int, legal_actions: List[Dict], phase: TurnPhase) -> dict:
        if phase == TurnPhase.END_TURN:
            return {"type": "end_turn"}
        if not legal_actions:
            return {"type": "end_main_action"}
        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        chosen_action = legal_actions[mapped_idx]
        if chosen_action.get("type") == "discard_cards" and "resources" not in chosen_action:
            return self._resolve_discard_action(chosen_action)
        return chosen_action

    def _decode_trade(self, action_idx: int, legal_actions: List[Dict], phase: TurnPhase) -> dict:
        if not self.enable_trading:
            if phase == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}
        if not legal_actions:
            if phase == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}
        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        return legal_actions[mapped_idx]

    def collect(self, policy, steps: int = 128) -> List[Dict[str, Any]]:
        storage: List[Dict[str, Any]] = []

        for _ in range(steps):
            # ---- Step A: Fetch all env states in parallel ----
            # One round-trip to all workers; returns (phase, legal_actions, obs_raw,
            # player_stats, pending_info) for every env simultaneously.
            all_state = self._vec_env.get_full_state()  # Dict[int, tuple]

            current_step_env_data: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
            gameplay_obs_list = []
            gameplay_env_indices = []
            trade_obs_list = []
            trade_env_indices = []
            auto_env_indices = []

            # ---- Step B: Build obs tensors (vectorized NumPy, main process) ----
            # This is now CPU-parallel across envs because obs building is done
            # after the parallel state fetch, and the Python loops here are over
            # feature *dimensions* (O(feature_dim)), not over game logic.
            for i in range(self.num_envs):
                phase, legal, obs_np, _player_stats, _pending_info = all_state[i]
                phase_name = self._phase_name(phase)
                current_step_env_data[i]["phase"] = phase_name
                current_step_env_data[i]["legal_actions"] = legal
                current_step_env_data[i]["env_phase"] = phase

                obs_tensor_dict = {
                    k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs_np.items()
                }

                if phase_name == "auto":
                    auto_env_indices.append(i)
                    current_step_env_data[i]["obs"] = obs_tensor_dict
                    current_step_env_data[i]["value"] = torch.tensor([[0.0]], device=self.device)
                else:
                    current_step_env_data[i]["obs"] = obs_tensor_dict

                    if phase_name == "gameplay":
                        gameplay_obs_list.append(obs_tensor_dict)
                        gameplay_env_indices.append(i)
                    else:
                        trade_obs_list.append(obs_tensor_dict)
                        trade_env_indices.append(i)

            # ---- Step C: Step auto-phase envs (no GPU needed) ----
            if auto_env_indices:
                auto_actions = {i: None for i in auto_env_indices}
                auto_results = self._vec_env.step(auto_actions)
                for i in auto_env_indices:
                    _obs_raw, reward, done, info = auto_results[i]
                    current_step_env_data[i]["reward"] = float(reward)
                    current_step_env_data[i]["done"] = bool(done)
                    current_step_env_data[i]["info"] = info
                    current_step_env_data[i]["env_action"] = None
                    if done:
                        self.game_stats_buffer.append({
                            "winner": info.get("winner") if info else None,
                            "total_steps": info.get("total_steps", 0) if info else 0,
                            "completed_naturally": info.get("completed_naturally", False) if info else False,
                        })

            # ---- Step D: Batch GPU inference for gameplay ----
            if gameplay_obs_list:
                batch_gameplay_obs = {
                    k: torch.cat([o[k] for o in gameplay_obs_list], dim=0)
                    for k in gameplay_obs_list[0].keys()
                }
                value, action_dict, log_prob_dict, tom_outputs = policy.act(
                    obs=batch_gameplay_obs, phase="gameplay", deterministic=False,
                )
                for j, env_idx in enumerate(gameplay_env_indices):
                    current_step_env_data[env_idx]["value"]       = value[j:j+1].detach().clone()
                    current_step_env_data[env_idx]["action"]      = {k: v[j:j+1].detach().clone() for k, v in action_dict.items()}
                    current_step_env_data[env_idx]["log_prob"]    = {k: v[j:j+1].detach().clone() for k, v in log_prob_dict.items()}
                    current_step_env_data[env_idx]["tom_outputs"] = {k: v[j:j+1].detach().clone() for k, v in tom_outputs.items()} if tom_outputs else {}

            # ---- Step E: Batch GPU inference for trade ----
            if trade_obs_list:
                batch_trade_obs = {
                    k: torch.cat([o[k] for o in trade_obs_list], dim=0)
                    for k in trade_obs_list[0].keys()
                }
                value, action_dict, log_prob_dict, tom_outputs = policy.act(
                    obs=batch_trade_obs, phase="trade", deterministic=False,
                )
                for j, env_idx in enumerate(trade_env_indices):
                    current_step_env_data[env_idx]["value"]       = value[j:j+1].detach().clone()
                    current_step_env_data[env_idx]["action"]      = {k: v[j:j+1].detach().clone() for k, v in action_dict.items()}
                    current_step_env_data[env_idx]["log_prob"]    = {k: v[j:j+1].detach().clone() for k, v in log_prob_dict.items()}
                    current_step_env_data[env_idx]["tom_outputs"] = {k: v[j:j+1].detach().clone() for k, v in tom_outputs.items()} if tom_outputs else {}

            # ---- Step F: Decode actions and step non-auto envs in parallel ----
            # Decode all actions first (main process, fast), then dispatch
            # all env.step() calls to workers simultaneously.
            env_actions_to_step: Dict[int, Any] = {}

            for i in gameplay_env_indices + trade_env_indices:
                env_data = current_step_env_data[i]
                phase_name = env_data["phase"]
                legal = env_data["legal_actions"]
                phase = env_data["env_phase"]

                if phase_name == "gameplay":
                    action_idx = int(env_data["action"]["gameplay_action"].detach().cpu().item())
                    env_action = self._decode_gameplay(action_idx, legal, phase)
                else:
                    action_idx = int(env_data["action"]["trade_action"].detach().cpu().item())
                    env_action = self._decode_trade(action_idx, legal, phase)

                env_data["env_action"] = env_action
                env_actions_to_step[i] = env_action

            # Dispatch all active envs to workers simultaneously
            if env_actions_to_step:
                step_results = self._vec_env.step(env_actions_to_step)
                for i, (_obs_raw, reward, done, info) in step_results.items():
                    current_step_env_data[i]["reward"] = float(reward)
                    current_step_env_data[i]["done"]   = bool(done)
                    current_step_env_data[i]["info"]   = info
                    if done:
                        self.game_stats_buffer.append({
                            "winner": info.get("winner") if info else None,
                            "total_steps": info.get("total_steps", 0) if info else 0,
                            "completed_naturally": info.get("completed_naturally", False) if info else False,
                        })

            # ---- Step G: Assemble storage records ----
            for i in range(self.num_envs):
                env_data = current_step_env_data[i]
                phase_name = env_data["phase"]

                if phase_name == "auto":
                    storage.append({
                        "obs":       {k: v.detach().clone() for k, v in env_data["obs"].items()},
                        "phase":     phase_name,
                        "action":    {
                            "gameplay_action": torch.tensor([-1], device=self.device),
                            "trade_action":    torch.tensor([-1], device=self.device),
                        },
                        "log_prob":  {
                            "gameplay_action": torch.tensor([0.0], device=self.device),
                            "trade_action":    torch.tensor([0.0], device=self.device),
                        },
                        "value":     env_data["value"],
                        "reward":    env_data.get("reward", 0.0),
                        "done":      env_data.get("done", False),
                        "info":      env_data.get("info", {}),
                        "env_action": None,
                        "tom_outputs": {},
                    })
                    continue

                tom = env_data.get("tom_outputs", {})
                storage.append({
                    "obs":       {k: v.detach().clone() for k, v in env_data["obs"].items()},
                    "phase":     phase_name,
                    "action":    {k: v.detach().clone() for k, v in env_data["action"].items()},
                    "log_prob":  {k: v.detach().clone() for k, v in env_data["log_prob"].items()},
                    "value":     env_data["value"].detach().clone(),
                    "reward":    env_data.get("reward", 0.0),
                    "done":      env_data.get("done", False),
                    "info":      env_data.get("info", {}),
                    "env_action": env_data.get("env_action"),
                    "tom_outputs": {k: v.detach().clone() for k, v in tom.items()} if tom else {},
                })

        return storage

    def close(self):
        """Shut down all worker processes cleanly."""
        self._vec_env.close()
