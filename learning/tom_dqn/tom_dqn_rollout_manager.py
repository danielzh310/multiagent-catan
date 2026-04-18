from __future__ import annotations

import sys
import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.constants import PlayerId, Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv


def _encode_gameplay_actions_batch(
    legal_actions: List[Dict],
    player_stats: Dict,
    max_actions: int,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (Placeholder) Encodes legal gameplay actions into a feature tensor.
    This is a placeholder implementation. The actual feature engineering logic
    from the original function should be placed here.
    """
    num_legal = len(legal_actions)
    features = np.zeros((max_actions, feature_dim), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=np.float32)

    # The model expects a mask of 1s for valid actions.
    # A real implementation would also populate the `features` array.
    if num_legal > 0:
        valid_indices = min(num_legal, max_actions)
        mask[:valid_indices] = 1.0

    return features, mask


def _encode_trade_actions_batch(
    legal_actions: List[Dict],
    player_stats: Dict,
    pending_info: Optional[Dict],
    max_actions: int,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (Placeholder) Encodes legal trade actions into a feature tensor.
    This is a placeholder implementation. The actual feature engineering logic
    from the original function should be placed here.
    """
    num_legal = len(legal_actions)
    features = np.zeros((max_actions, feature_dim), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=np.float32)

    if num_legal > 0:
        valid_indices = min(num_legal, max_actions)
        mask[:valid_indices] = 1.0

    return features, mask

def _worker_build_obs(
    obs_raw: Dict,
    legal_actions: List[Dict],
    player_stats: Dict,
    pending_info: Optional[Dict],
    phase: TurnPhase,
) -> Dict[str, np.ndarray]:
    """
    Builds the obs numpy array dict from pre-fetched state data.
    Enhanced with global state for ToM capabilities.
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

    def build_global_state() -> np.ndarray:
        opponent_vectors = []
        for opp in other_players[:3]:
            opponent_vectors.append(to_vec(opp))
        while len(opponent_vectors) < 3:
            opponent_vectors.append(np.zeros(64, dtype=np.float32))
        opponent_flat = np.concatenate(opponent_vectors, axis=0)

        resource_bank = obs_raw.get("game", {}).get("resource_bank", {})
        bank_vec = np.zeros(5, dtype=np.float32)
        for idx, res in enumerate(["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]):
            bank_vec[idx] = float(resource_bank.get(res, 0)) / 19.0

        pending_vec = np.zeros(4, dtype=np.float32)
        if pending_info is not None:
            pending_vec[0] = 1.0
            pending_vec[1] = float(pending_info.get("counter_count", 0)) / 3.0
            pending_vec[2] = float(pending_info.get("proposer", 0)) / 3.0
            pending_vec[3] = float(pending_info.get("target", 0)) / 3.0

        return np.concatenate([self_np, opponent_flat, bank_vec, pending_vec], axis=0)

    global_np = build_global_state()

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
        ToMEnhancedDQNRolloutManager.MAX_GAMEPLAY_ACTIONS, ToMEnhancedDQNRolloutManager.GAMEPLAY_FEATURE_DIM,
    )
    trade_np, trade_mask_np = _encode_trade_actions_batch(
        legal_actions, player_stats, pending_info,
        ToMEnhancedDQNRolloutManager.MAX_TRADE_ACTIONS, ToMEnhancedDQNRolloutManager.TRADE_FEATURE_DIM,
    )

    return {
        "board": board_np, "self": self_np, "opponent": op_np,
        "global_state": global_np,
        "gameplay_candidates": gameplay_np, "gameplay_mask": gameplay_mask_np,
        "trade_candidates": trade_np, "trade_mask": trade_mask_np,
    }


def _load_tom_dqn_opponent_policies(opponent_paths: List[str], worker_id: int) -> List['ToMEnhancedDQNPolicy']:
    """Loads a list of ToMEnhancedDQNPolicy opponents from checkpoint paths."""
    from learning.tom_dqn.tom_dqn_policy import ToMEnhancedDQNPolicy
    opponent_policies = []
    if not opponent_paths:
        return opponent_policies

    for path in opponent_paths:
        try:
            # Opponents run on CPU to save GPU memory
            ckpt = torch.load(path, map_location="cpu")
            opp_policy = ToMEnhancedDQNPolicy(device="cpu")
            # Check for different keys from different trainers
            if "policy_state_dict" in ckpt:
                state_dict = ckpt["policy_state_dict"]
            elif "policy" in ckpt:
                state_dict = ckpt["policy"]
            else:
                state_dict = ckpt
            opp_policy.load_state_dict(state_dict, strict=False)
            opp_policy.eval()
            opponent_policies.append(opp_policy)
        except Exception as e:
            print(f"Worker {worker_id} failed to load opponent {path}: {e}", file=sys.stderr)
    return opponent_policies

def _worker(
    worker_id: int,
    env_indices: List[int],
    cmd_pipe: Connection,
    enable_trading: bool,
    max_steps: Optional[int],
    opponent_paths: Optional[List[str]] = None,
) -> None:
    """
    Runs a subset of CatanEnv instances in a separate process.
    Enhanced with global state observation building.
    """
    # Load opponent policies from the provided paths.
    opponent_policies = _load_tom_dqn_opponent_policies(opponent_paths or [], worker_id)
    # Build only the envs owned by this worker
    envs: Dict[int, CatanEnv] = {}
    for idx in env_indices:
        envs[idx] = CatanEnv(
            enable_trading=enable_trading, max_steps=max_steps, opponent_policies=opponent_policies
        )
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

            elif cmd == "get_full_state":
                results = []
                for idx in payload:
                    phase = envs[idx].get_phase()
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
                    results.append((idx, phase, legal, envs[idx].get_observation(), player_stats, pending_info))
                cmd_pipe.send(results)

            elif cmd == "update_opponents":
                new_opponent_paths = payload
                new_opponent_policies = _load_tom_dqn_opponent_policies(new_opponent_paths, worker_id)
                for env in envs.values():
                    env.update_opponent_policies(new_opponent_policies)

            elif cmd == "close":
                break

    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
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
        opponent_paths: Optional[List[str]] = None,
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
                args=(w, self._worker_env_indices[w], child_conn, enable_trading, max_steps, opponent_paths),
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

    def update_opponents(self, opponent_paths: List[str]):
        """Sends a command to all workers to update their opponent policies."""
        for pipe in self._pipes:
            pipe.send(("update_opponents", opponent_paths))

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


class ToMEnhancedDQNRolloutManager:
    MAX_GAMEPLAY_ACTIONS = 256
    GAMEPLAY_FEATURE_DIM = 40
    MAX_TRADE_ACTIONS = 128
    TRADE_FEATURE_DIM = 32
    GLOBAL_FEATURE_DIM = 265

    def __init__(
        self,
        num_envs: int = 8,
        num_workers: int = 4,
        device: str = "cpu",
        enable_trading: bool = True,
        max_steps: Optional[int] = None,
        opponent_paths: Optional[List[str]] = None,
    ):
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.device = device
        self.enable_trading = enable_trading
        self.game_stats_buffer: List[Dict] = []

        # Async vector env: all Catan logic runs in parallel worker processes
        self._vec_env = AsyncVectorEnv(
            num_envs=num_envs,
            num_workers=num_workers,
            enable_trading=enable_trading,
            max_steps=max_steps,
            opponent_paths=opponent_paths,
        )

    def update_opponents(self, opponent_paths: List[str]):
        """Sends new opponent paths to all worker environments."""
        self._vec_env.update_opponents(opponent_paths)

    def collect(self, policy, epsilon_scheduler, steps: int = 128) -> List[Dict[str, Any]]:
        storage: List[Dict[str, Any]] = []

        for _ in range(steps):
            # ---- Step A: Fetch all env states in parallel ----
            all_state = self._vec_env.get_full_state()  # Dict[int, tuple]

            current_step_env_data: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
            gameplay_obs_list = []
            gameplay_env_indices = []
            trade_obs_list = []
            trade_env_indices = []
            auto_env_indices = []

            # ---- Step B: Build obs tensors (vectorized NumPy, main process) ----
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
                epsilon = epsilon_scheduler.get_epsilon()
                value, action_dict, log_prob_dict, tom_outputs = policy.act(
                    obs=batch_gameplay_obs, phase="gameplay", epsilon=epsilon,
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
                epsilon = epsilon_scheduler.get_epsilon()
                value, action_dict, log_prob_dict, tom_outputs = policy.act(
                    obs=batch_trade_obs, phase="trade", epsilon=epsilon,
                )
                for j, env_idx in enumerate(trade_env_indices):
                    current_step_env_data[env_idx]["value"]       = value[j:j+1].detach().clone()
                    current_step_env_data[env_idx]["action"]      = {k: v[j:j+1].detach().clone() for k, v in action_dict.items()}
                    current_step_env_data[env_idx]["log_prob"]    = {k: v[j:j+1].detach().clone() for k, v in log_prob_dict.items()}
                    current_step_env_data[env_idx]["tom_outputs"] = {k: v[j:j+1].detach().clone() for k, v in tom_outputs.items()} if tom_outputs else {}

            # ---- Step F: Decode actions and step non-auto envs in parallel ----
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

            # ---- Step G: Step environments and collect results ----
            if env_actions_to_step:
                step_results = self._vec_env.step(env_actions_to_step)
                for i, (_obs_raw, reward, done, info) in step_results.items():
                    env_data = current_step_env_data[i]
                    env_data["reward"] = float(reward)
                    env_data["done"] = bool(done)
                    env_data["info"] = info

                    if done:
                        self.game_stats_buffer.append({
                            "winner": info.get("winner") if info else None,
                            "total_steps": info.get("total_steps", 0) if info else 0,
                            "completed_naturally": info.get("completed_naturally", False) if info else False,
                        })

            # ---- Step H: Assemble storage records ----
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

    def _phase_name(self, phase: TurnPhase) -> str:
        if phase.value in [1, 2]:  # Corresponds to GAMEPLAY and INITIAL_PLACEMENT
            return "gameplay"
        elif phase.value in [3, 4]:  # Corresponds to TRADE_PROPOSE and TRADE_RESPOND
            return "trade"
        else:
            return "auto"

    def _decode_gameplay(self, action_idx: int, legal_actions: List[Dict], phase: TurnPhase) -> dict:
        if not legal_actions:
            return {"type": "end_turn"}
        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        chosen_action = legal_actions[mapped_idx]
        if chosen_action.get("type") == "discard_cards" and "resources" not in chosen_action:
            return self._resolve_discard_action(chosen_action)
        return chosen_action

    def _decode_trade(self, action_idx: int, legal_actions: List[Dict], phase: TurnPhase) -> dict:
        if not self.enable_trading:
            if phase.value == 4:  # Corresponds to TRADE_RESPOND
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}
        if not legal_actions:
            if phase.value == 4:  # Corresponds to TRADE_RESPOND
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

    def close(self):
        self._vec_env.close()