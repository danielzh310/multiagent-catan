from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.constants import PlayerId, Resource, DevCard
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv


def _resource_slot(resource_value: Any) -> int:
    if resource_value is None:
        return -1
    try:
        return int(Resource(resource_value))
    except (ValueError, TypeError):
        return -1


def _dev_card_slot(card_value: Any) -> int:
    if card_value is None:
        return -1
    try:
        return int(DevCard(card_value))
    except (ValueError, TypeError):
        return -1


def _encode_gameplay_action(action: Dict[str, Any], phase: TurnPhase, player_stats: Dict) -> List[float]:
    features = [0.0] * PPORolloutManager.GAMEPLAY_FEATURE_DIM
    action_type = action.get("type", "")
    action_types = {
        "build_settlement": 0, "build_road": 1, "build_city": 2,
        "buy_dev_card": 3, "play_dev_card": 4, "bank_trade": 5,
        "move_robber": 6, "discard_cards": 7, "end_main_action": 8, "end_turn": 9
    }
    if (action_type_idx := action_types.get(action_type)) is not None:
        features[action_type_idx] = 1.0

    if "connection" in action: features[10] = float(action["connection"]) / 128.0
    if "tile" in action: features[11] = float(action["tile"]) / 19.0
    if "connection_1" in action and action["connection_1"] is not None: features[12] = float(action["connection_1"]) / 128.0
    if "connection_2" in action and action["connection_2"] is not None: features[13] = float(action["connection_2"]) / 128.0

    if (give_slot := _resource_slot(action.get("give"))) >= 0: features[14 + give_slot] = 1.0
    if (receive_slot := _resource_slot(action.get("receive"))) >= 0: features[19 + receive_slot] = 1.0
    if (resource_slot := _resource_slot(action.get("resource"))) >= 0: features[24] = float(resource_slot + 1) / 5.0
    if (resource_1_slot := _resource_slot(action.get("resource_1"))) >= 0: features[25] = float(resource_1_slot + 1) / 5.0
    if (resource_2_slot := _resource_slot(action.get("resource_2"))) >= 0: features[26] = float(resource_2_slot + 1) / 5.0
    if (card_slot := _dev_card_slot(action.get("card"))) >= 0: features[27] = float(card_slot + 1) / 5.0
    if (rate := action.get("rate")) is not None: features[28] = float(rate) / 4.0
    if (required := action.get("required")) is not None: features[29] = float(required) / 8.0
    if isinstance(resources_to_discard := action.get("resources"), dict):
        features[30] = sum(float(v) for v in resources_to_discard.values()) / 8.0
        features[31] = sum(1 for v in resources_to_discard.values() if int(v) > 0) / 5.0
    if action.get("type") == "play_dev_card": features[32] = 1.0

    # Add phase information
    features[33] = float(phase.value) / 10.0
    features[34] = 1.0 if phase == TurnPhase.SETUP else 0.0
    features[35] = 1.0 if phase == TurnPhase.MAIN_ACTION else 0.0
    features[36] = 1.0 if phase == TurnPhase.TRADE_PROPOSE else 0.0
    features[37] = 1.0 if phase == TurnPhase.TRADE_RESPOND else 0.0

    # Add player stats
    features[38] = float(player_stats["vp"]) / 10.0
    features[39] = float(player_stats["resource_total"]) / 20.0

    return features


def _encode_trade_action(action: Dict[str, Any], player_stats: Dict, pending_info: Optional[Dict]) -> List[float]:
    features = [0.0] * PPORolloutManager.TRADE_FEATURE_DIM
    action_type = action.get("type", "")
    action_types = {"skip_trade": 0, "propose_trade": 1, "accept_trade": 2, "reject_trade": 3, "counter_trade": 4}
    if (action_type_idx := action_types.get(action_type)) is not None:
        features[action_type_idx] = 1.0

    features[5] = float(player_stats["vp"]) / 10.0
    features[6] = float(player_stats["resource_total"]) / 20.0

    if pending_info is not None:
        features[7] = 1.0
        features[8] = float(pending_info.get("counter_count", 0)) / 3.0
        features[9] = float(pending_info.get("proposer", 0)) / 3.0
        features[10] = float(pending_info.get("target", 0)) / 3.0

    if (target := action.get("target")) is not None:
        try:
            features[11 + int(target)] = 1.0
        except (ValueError, TypeError, IndexError):
            pass

    offer = action.get("offer") or action.get("counter_offer") or {}
    for i, res in enumerate(["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]):
        features[15 + i] = float(offer.get(res, 0)) / 5.0

    request = action.get("request") or action.get("counter_request") or {}
    for i, res in enumerate(["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]):
        features[20 + i] = float(request.get(res, 0)) / 5.0

    if (response_type := action.get("response_type")) is not None:
        response_types = {"accept": 25, "reject": 26, "counter": 27}
        if (response_idx := response_types.get(response_type)) is not None:
            features[response_idx] = 1.0

    return features


def _encode_gameplay_actions_batch(
    legal_actions: List[Dict],
    player_stats: Dict,
    phase: TurnPhase,
    max_actions: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    candidates = np.zeros((max_actions, feature_dim), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=bool)

    capped_actions = legal_actions[:max_actions]
    for idx, action in enumerate(capped_actions):
        candidates[idx] = np.array(_encode_gameplay_action(action, phase, player_stats), dtype=np.float32)
        mask[idx] = True
    if not capped_actions: mask[0] = True
    return candidates, mask


def _encode_trade_actions_batch(
    legal_actions: List[Dict],
    player_stats: Dict,
    pending_info: Optional[Dict],
    max_actions: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    candidates = np.zeros((max_actions, feature_dim), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=bool)

    capped_actions = legal_actions[:max_actions]
    for idx, action in enumerate(capped_actions):
        candidates[idx] = np.array(_encode_trade_action(action, player_stats, pending_info), dtype=np.float32)
        mask[idx] = True
    if not capped_actions: mask[0] = True
    return candidates, mask


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
        legal_actions, player_stats, phase,
        PPORolloutManager.MAX_GAMEPLAY_ACTIONS, PPORolloutManager.GAMEPLAY_FEATURE_DIM,
    )
    trade_np, trade_mask_np = _encode_trade_actions_batch(
        legal_actions, player_stats, pending_info,
        PPORolloutManager.MAX_TRADE_ACTIONS, PPORolloutManager.TRADE_FEATURE_DIM,
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
                        "resource_total": float(sum(int(v) for v in player.resources.values())),
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

            elif cmd == "close":
                break

    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        traceback.print_exc()


class PPORolloutManager:
    MAX_GAMEPLAY_ACTIONS = 256
    GAMEPLAY_FEATURE_DIM = 40
    MAX_TRADE_ACTIONS = 128
    TRADE_FEATURE_DIM = 32

    def __init__(
        self,
        num_envs: int = 8,
        num_workers: int = 4,
        device: str = "cpu",
        enable_trading: bool = True,
        max_steps: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.num_workers = min(num_workers, num_envs)
        self.device = device
        self.enable_trading = enable_trading

        # Distribute envs across workers
        self._env_to_worker = {}
        self._worker_env_indices = [[] for _ in range(self.num_workers)]
        envs_per_worker = num_envs // self.num_workers
        extra_envs = num_envs % self.num_workers

        env_idx = 0
        for w in range(self.num_workers):
            num_envs_this_worker = envs_per_worker + (1 if w < extra_envs else 0)
            for _ in range(num_envs_this_worker):
                self._env_to_worker[env_idx] = w
                self._worker_env_indices[w].append(env_idx)
                env_idx += 1

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
            self._pipes.append(parent_conn)  # type: ignore
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

    def collect(self, policy, steps: int = 128) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """
        Collect rollouts for PPO training.
        Returns a tuple of:
          - storage: List of transition dictionaries.
          - next_value: The value of the state after the last step, for GAE.
        """
        storage: List[Dict[str, Any]] = []

        for _ in range(steps):
            # Get current states from all environments
            all_state = self.get_full_state()

            # Prepare actions for all environments
            env_actions = {}
            current_storage = [{} for _ in range(self.num_envs)]

            # Group by phase for batching
            gameplay_indices = []
            gameplay_obs_list = []
            trade_indices = []
            trade_obs_list = []

            for env_idx in range(self.num_envs):
                phase, legal_actions, obs_np, player_stats, pending_info = all_state[env_idx]
                phase_name = self._phase_name(phase)

                # obs_np is already built by worker
                obs_tensor = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs_np.items()}

                current_storage[env_idx] = {
                    "obs": obs_tensor,
                    "phase": phase_name,
                    "phase_obj": phase,
                    "legal_actions": legal_actions,
                }

                if phase_name == "auto":
                    # Auto phase - just step with no action
                    env_actions[env_idx] = None
                else:
                    # Collect for batching
                    if phase_name == "gameplay":
                        gameplay_indices.append(env_idx)
                        gameplay_obs_list.append(obs_tensor)
                    else:  # trade
                        trade_indices.append(env_idx)
                        trade_obs_list.append(obs_tensor)

            # Batch policy calls for gameplay
            if gameplay_obs_list:
                batch_gameplay_obs = {
                    k: torch.cat([obs[k] for obs in gameplay_obs_list], dim=0)
                    for k in gameplay_obs_list[0].keys()
                }
                batch_values, batch_actions, batch_log_probs = policy.act(
                    obs=batch_gameplay_obs, phase="gameplay", deterministic=False
                )

                # Split results back
                for i, env_idx in enumerate(gameplay_indices):
                    action_dict = {
                        "gameplay_action": batch_actions["gameplay_action"][i:i+1],
                        "trade_action": batch_actions["trade_action"][i:i+1],
                    }
                    log_prob_dict = {
                        "gameplay_action": batch_log_probs["gameplay_action"][i:i+1],
                        "trade_action": batch_log_probs["trade_action"][i:i+1],
                    }
                    value = batch_values[i:i+1]

                    legal_actions = current_storage[env_idx]["legal_actions"]
                    phase = current_storage[env_idx]["phase_obj"]
                    action_idx = int(action_dict["gameplay_action"].item())
                    env_action = self._decode_gameplay(action_idx, legal_actions, phase)

                    env_actions[env_idx] = env_action
                    current_storage[env_idx].update({
                        "action": action_dict,
                        "log_prob": log_prob_dict,
                        "value": value,
                        "env_action": env_action,
                    })

            # Batch policy calls for trade
            if trade_obs_list:
                batch_trade_obs = {
                    k: torch.cat([obs[k] for obs in trade_obs_list], dim=0)
                    for k in trade_obs_list[0].keys()
                }
                batch_values, batch_actions, batch_log_probs = policy.act(
                    obs=batch_trade_obs, phase="trade", deterministic=False
                )

                # Split results back
                for i, env_idx in enumerate(trade_indices):
                    action_dict = {
                        "gameplay_action": batch_actions["gameplay_action"][i:i+1],
                        "trade_action": batch_actions["trade_action"][i:i+1],
                    }
                    log_prob_dict = {
                        "gameplay_action": batch_log_probs["gameplay_action"][i:i+1],
                        "trade_action": batch_log_probs["trade_action"][i:i+1],
                    }
                    value = batch_values[i:i+1]

                    legal_actions = current_storage[env_idx]["legal_actions"]
                    phase = current_storage[env_idx]["phase_obj"]
                    action_idx = int(action_dict["trade_action"].item())
                    env_action = self._decode_trade(action_idx, legal_actions, phase)

                    env_actions[env_idx] = env_action
                    current_storage[env_idx].update({
                        "action": action_dict,
                        "log_prob": log_prob_dict,
                        "value": value,
                        "env_action": env_action,
                    })

            # Step all environments at once
            step_results = self.step(env_actions)

            # Process results
            for env_idx in range(self.num_envs):
                env_data = current_storage[env_idx]
                phase_name = env_data["phase"]

                obs_raw, reward, done, info = step_results[env_idx]

                if phase_name == "auto":
                    storage.append({
                        "obs": env_data["obs"],
                        "phase": phase_name,
                        "action": {
                            "gameplay_action": torch.tensor([[-1]], dtype=torch.long, device=self.device),
                            "trade_action": torch.tensor([[-1]], dtype=torch.long, device=self.device),
                        },
                        "log_prob": {
                            "gameplay_action": torch.tensor([[0.0]], device=self.device),
                            "trade_action": torch.tensor([[0.0]], device=self.device),
                        },
                        "value": torch.tensor([[0.0]], device=self.device),
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                        "env_action": None,
                    })
                else:
                    storage.append({
                        "obs": env_data["obs"],
                        "phase": env_data["phase"],
                        "action": env_data["action"],
                        "log_prob": env_data["log_prob"],
                        "value": env_data["value"],
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                        "env_action": env_data["env_action"],
                    })

        # After the main loop, compute the value for the next state for GAE
        next_value = torch.zeros(self.num_envs, 1, device=self.device)
        final_full_state = self.get_full_state()

        gameplay_indices = []
        gameplay_obs_list = []
        trade_indices = []
        trade_obs_list = []

        for env_idx in range(self.num_envs):
            phase, _, obs_np, _, _ = final_full_state[env_idx]
            phase_name = self._phase_name(phase)
            obs_tensor = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs_np.items()}

            if phase_name == "gameplay":
                gameplay_indices.append(env_idx)
                gameplay_obs_list.append(obs_tensor)
            elif phase_name == "trade":
                trade_indices.append(env_idx)
                trade_obs_list.append(obs_tensor)

        with torch.no_grad():
            if gameplay_obs_list:
                batch_obs = {k: torch.cat([o[k] for o in gameplay_obs_list]) for k in gameplay_obs_list[0]}
                values = policy.get_value(batch_obs, "gameplay")
                for i, env_idx in enumerate(gameplay_indices):
                    next_value[env_idx] = values[i]

            if trade_obs_list:
                batch_obs = {k: torch.cat([o[k] for o in trade_obs_list]) for k in trade_obs_list[0]}
                values = policy.get_value(batch_obs, "trade")
                for i, env_idx in enumerate(trade_indices):
                    next_value[env_idx] = values[i]

        return storage, next_value

    def _phase_name(self, phase: TurnPhase) -> str:
        if phase in [TurnPhase.SETUP, TurnPhase.MAIN_ACTION, TurnPhase.END_TURN]:
            return "gameplay"
        elif phase in [TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND]:
            return "trade"
        else:
            return "auto"

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

    def _resolve_discard_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        required = int(action.get("required", 0))
        available = action.get("available", {})
        ordered_resources = sorted(available.items(), key=lambda item: (-int(item[1]), item[0]))
        resources_to_discard: Dict[Resource, int] = {r: 0 for r in Resource}
        remaining = required
        for resource_name, count in ordered_resources:
            if remaining <= 0: break
            take = min(int(count), remaining)
            if take > 0:
                resources_to_discard[Resource[resource_name]] = take
                remaining -= take
        if remaining > 0: return action
        resolved = dict(action)
        resolved["resources"] = resources_to_discard
        return resolved