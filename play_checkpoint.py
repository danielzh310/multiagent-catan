import argparse
import os
import torch

from core.constants import Resource
from learning.unified.unified_policy import UnifiedPolicy
from learning.unified.unified_rollout_manager import UnifiedRolloutManager


def parse_args():
    parser = argparse.ArgumentParser(description="Play one game from a unified checkpoint and print actions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy deterministic actions")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation steps")
    parser.add_argument("--show-obs", action="store_true", help="Print raw observation vectors each step")
    parser.add_argument("--show-board", action="store_true", help="Print board state each step")
    parser.add_argument("--gameplay-only", action="store_true", help="Disable trading and only evaluate gameplay")
    return parser.parse_args()


def snapshot_resources(env):
    return {
        str(p): {k.name: int(v) for k, v in env.engine.players[p].resources.items()}
        for p in env.engine.players
    }


def snapshot_vp(env):
    return {
        str(p): int(env.engine.players[p].update_victory_points())
        for p in env.engine.players
    }


def diff_resources(before, after):
    out = {}
    for player, res_map in after.items():
        delta_map = {}
        for resource, value_after in res_map.items():
            value_before = before[player][resource]
            delta = int(value_after) - int(value_before)
            if delta != 0:
                delta_map[resource] = delta
        if delta_map:
            out[player] = delta_map
    return out


def format_ascii_board(env):
    def fmt_tile(t_id):
        tile = env.engine.board.tiles[t_id]
        res = tile.resource.name[:2]
        rob = "*" if getattr(tile, "has_robber", False) else " "
        num = str(tile.number) if tile.number is not None else ""
        return f"[{res}{rob}{num:>2}]"

    r0 = [7, 12, 16]
    r1 = [3, 8, 13, 17]
    r2 = [0, 4, 9, 14, 18]
    r3 = [1, 5, 10, 15]
    r4 = [2, 6, 11]

    lines = [
        "ASCII BOARD:",
        " " * 10 + "   ".join(fmt_tile(i) for i in r0),
        " " * 5 + "   ".join(fmt_tile(i) for i in r1),
        "" + "   ".join(fmt_tile(i) for i in r2),
        " " * 5 + "   ".join(fmt_tile(i) for i in r3),
        " " * 10 + "   ".join(fmt_tile(i) for i in r4),
    ]
    return "\n".join(lines)


def format_board_state(env):
    lines = []
    lines.append("=== CURRENT BOARD STATE ===")
    lines.append(format_ascii_board(env))
    lines.append("")
    lines.append(f"phase={env.get_phase().name}")
    lines.append(f"current_player={env.get_current_player_id()}")
    lines.append(f"last_roll={env.get_last_roll()}")
    lines.append(f"last_robber_event={env.get_last_robber_event()}")
    lines.append(f"board: {len(env.engine.board.tiles)} tiles, {len(env.engine.board.vertices)} vertices, {len(env.engine.board.connections)} connections")
    lines.append("")

    # Show tiles
    lines.append("TILES:")
    for i, tile in enumerate(env.engine.board.tiles):
        robber_str = " (ROBBER)" if tile.has_robber else ""
        lines.append(f"  tile_{i}: {tile.resource.name}@{tile.number}{robber_str}")
    lines.append("")

    # Show tile vertex mapping
    lines.append("TILE VERTICES:")
    lines.append("  (Corner order: [BottomRight, TopRight, Top, TopLeft, BottomLeft, Bottom])")
    lines.append("       v2(Top)      ")
    lines.append("   v3          v1   ")
    lines.append("       [Tile]       ")
    lines.append("   v4          v0   ")
    lines.append("      v5(Bottom)    ")
    for i, tile in enumerate(env.engine.board.tiles):
        v_ids = [f"{v.id:02d}" for v in tile.vertices]
        lines.append(f"  tile_{i:>2} ({tile.resource.name[:2]:>2}): [{', '.join(v_ids)}]")
    lines.append("")

    # Show ports
    lines.append("PORTS:")
    for port in env.engine.board.ports:
        v1, v2 = port.vertices
        shared_tiles = set(t.id for t in v1.tiles) & set(t.id for t in v2.tiles)
        tile_id = list(shared_tiles)[0] if shared_tiles else None
        res_str = "3:1" if port.resource is None else f"{port.resource.name} 2:1"
        lines.append(f"  P{port.id}: {res_str} at V{v1.id}-V{v2.id} (Tile {tile_id})")
    lines.append("")

    # Show vertices and their buildings
    lines.append("VERTICES (settlements/cities):")
    occupied_vertices = []
    for i in range(len(env.engine.board.vertices)):  # Show all vertices
        vertex = env.engine.board.vertices[i]
        owner = vertex.owner()
        building = "empty"
        if owner is not None and vertex.building is not None:
            if vertex.building.type.name == "SETTLEMENT":
                building = f"settlement({owner.name})"
                occupied_vertices.append(f"V{i}: {building}")
            elif vertex.building.type.name == "CITY":
                building = f"city({owner.name})"
                occupied_vertices.append(f"V{i}: {building}")
    if occupied_vertices:
        for line in occupied_vertices:
            lines.append(f"  {line}")
    else:
        lines.append("  (no buildings placed)")
    lines.append("")

    # Show connections and roads
    lines.append("CONNECTIONS (roads):")
    roads = []
    for i in range(len(env.engine.board.connections)):  # Show all connections
        conn = env.engine.board.connections[i]
        if conn.owner is not None:
            v1_id = env.engine.board.vertices.index(conn.v1)
            v2_id = env.engine.board.vertices.index(conn.v2)
            roads.append(f"C{i} (V{v1_id}-V{v2_id}): {conn.owner.name}")
    if roads:
        for line in roads:
            lines.append(f"  {line}")
    else:
        lines.append("  (no roads placed)")
    lines.append("")

    # Show player states
    lines.append("PLAYERS:")
    for player in env.engine.player_order:
        state = env.engine.players[player]
        resources = (
            f"WOOD:{state.resources[Resource.WOOD]}, "
            f"BRICK:{state.resources[Resource.BRICK]}, "
            f"SHEEP:{state.resources[Resource.SHEEP]}, "
            f"WHEAT:{state.resources[Resource.WHEAT]}, "
            f"ORE:{state.resources[Resource.ORE]}"
        )
        lines.append(
            f"  {player.name}: settlements={state.n_settlements}, "
            f"cities={state.n_cities}, roads={state.n_roads}, "
            f"vp={state.update_victory_points()}, resources={{ {resources} }}"
        )

    return "\n".join(lines)


def load_state_dict_from_checkpoint(checkpoint_path, device):
    raw = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if "policy" in raw:
            return raw["policy"]
        if "models" in raw and isinstance(raw["models"], dict):
            if "model" in raw["models"]:
                return raw["models"]["model"]
        candidate = raw
        if all(isinstance(k, str) for k in candidate.keys()):
            return candidate

    raise ValueError(f"Unable to locate state_dict in checkpoint: {checkpoint_path}")


def run_single_unified_game(
    model,
    checkpoint_path,
    device,
    deterministic=False,
    max_steps=2000,
    show_obs=False,
    show_board=False,
    gameplay_only=False,
):
    model.eval()
    env_manager = UnifiedRolloutManager(num_envs=1, device=device, enable_trading=not gameplay_only)
    env = env_manager.envs[0]

    report = []
    total_steps = 0
    done = False

    while not done and total_steps < max_steps:
        phase = env.get_phase()
        phase_name = (
            "gameplay"
            if phase.name in ("SETUP", "MAIN_ACTION", "END_TURN")
            else "trade"
            if phase.name in ("TRADE_PROPOSE", "TRADE_RESPOND")
            else "auto"
        )

        resources_before = snapshot_resources(env)
        vp_before = snapshot_vp(env)
        obs_before = env.get_observation()
        legal_before = obs_before.get("legal_actions", [])

        if phase_name == "auto":
            next_obs, reward, done, info = env.step(None)

            last_roll = next_obs["game"].get("last_roll")
            robber_event = next_obs["game"].get("last_robber_event")

            print(
                f"step={total_steps} phase={phase.name} player={env.get_current_player_id()} "
                f"action=auto roll={last_roll} reward={reward:.3f} done={done}"
            )
            if robber_event is not None:
                print(f" robber_event={robber_event}")

            resources_after = snapshot_resources(env)
            vp_after = snapshot_vp(env)
            resource_delta = diff_resources(resources_before, resources_after)

            if resource_delta:
                print(f" resource_delta={resource_delta}")
            if vp_after != vp_before:
                print(f" vp={vp_after}")

            report.append(
                {
                    "step": total_steps,
                    "phase": phase.name,
                    "action": None,
                    "legal_actions_before": legal_before,
                    "roll": last_roll,
                    "robber_event": robber_event,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "vp_before": vp_before,
                    "vp_after": vp_after,
                    "resources_before": resources_before,
                    "resources_after": resources_after,
                    "resource_delta": resource_delta,
                }
            )
            total_steps += 1
            continue

        current_player = env.get_current_player_id()
        obs = env_manager._build_obs(env)

        if show_board:
            print(format_board_state(env))
            print()

        value, action_dict, _, tom_out = model.act(
            obs=obs,
            phase=phase_name,
            deterministic=deterministic,
        )

        if gameplay_only and phase_name == "trade":
            env_action = (
                {"type": "reject_trade", "response_type": "reject"}
                if phase.name == "TRADE_RESPOND"
                else {"type": "skip_trade"}
            )
        elif phase_name == "gameplay":
            env_action = env_manager._decode_gameplay(int(action_dict["gameplay_action"].item()), env)
        else:
            env_action = env_manager._decode_trade(int(action_dict["trade_action"].item()), env)

        next_obs, reward, done, info = env.step(env_action)

        last_roll = next_obs["game"].get("last_roll")
        robber_event = next_obs["game"].get("last_robber_event")
        setup_stage = next_obs["game"].get("initial_placement_stage")
        setup_phase = next_obs["game"].get("initial_placement_phase")

        resources_after = snapshot_resources(env)
        vp_after = snapshot_vp(env)
        resource_delta = diff_resources(resources_before, resources_after)

        print(f"step={total_steps} phase={phase.name} player={current_player}")
        print(f" legal_actions_before={legal_before}")
        print(f" env_action={env_action}")
        print(
            " action_dict={"
            + ", ".join(f"{k}:{v.detach().cpu().numpy().tolist()}" for k, v in action_dict.items())
            + "}"
        )
        print(f" roll={last_roll}")
        print(f" setup_phase={setup_phase} setup_stage={setup_stage}")
        if robber_event is not None:
            print(f" robber_event={robber_event}")
        print(f" reward={reward:.3f} done={done}")
        print(f" value={float(value.item()) if hasattr(value, 'item') else None}")
        print(f" vp_before={vp_before}")
        print(f" vp_after={vp_after}")
        if resource_delta:
            print(f" resource_delta={resource_delta}")
        print(f" resources_after={resources_after}\n")

        report.append(
            {
                "step": total_steps,
                "phase": phase.name,
                "player": str(current_player),
                "legal_actions_before": legal_before,
                "action_dict": {k: v.detach().cpu().numpy().tolist() for k, v in action_dict.items()},
                "env_action": env_action,
                "roll": last_roll,
                "setup_phase": setup_phase,
                "setup_stage": setup_stage,
                "robber_event": robber_event,
                "reward": reward,
                "done": done,
                "value": float(value.item()) if hasattr(value, "item") else None,
                "tom": {k: v.detach().cpu().numpy().tolist() for k, v in tom_out.items()} if tom_out is not None else None,
                "vp_before": vp_before,
                "vp_after": vp_after,
                "resources_before": resources_before,
                "resources_after": resources_after,
                "resource_delta": resource_delta,
            }
        )

        total_steps += 1

    winner = env.engine.winner
    stats = {
        "winner": str(winner) if winner is not None else None,
        "victory_points": snapshot_vp(env),
        "total_steps": total_steps,
        "done": done,
        "report": report,
        "final_board_state": format_board_state(env),
    }
    return stats


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = UnifiedPolicy().to(device)
    state_dict = load_state_dict_from_checkpoint(args.checkpoint, device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict

        stripped = OrderedDict()
        for k, v in state_dict.items():
            stripped[k.replace("module.", "")] = v
        model.load_state_dict(stripped)

    stats = run_single_unified_game(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        show_obs=args.show_obs,
        show_board=args.show_board,
        gameplay_only=args.gameplay_only,
    )

    print("\n=== GAME RESULT ===")
    print(f"winner={stats['winner']}")
    print(f"victory_points={stats['victory_points']}")
    print(f"total_steps={stats['total_steps']}")
    print(f"done={stats['done']}")
    print(stats["final_board_state"])

    os.makedirs("gameplay", exist_ok=True)
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    suffix = "_gameplay_only" if args.gameplay_only else ""
    out_path = os.path.join("gameplay", f"gameplay_{checkpoint_name}{suffix}_steps{stats['total_steps']}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== GAME RESULT ===\n")
        f.write(f"winner={stats['winner']}\n")
        f.write(f"victory_points={stats['victory_points']}\n")
        f.write(f"total_steps={stats['total_steps']}\n")
        f.write(f"done={stats['done']}\n\n")
        f.write(stats["final_board_state"])
        f.write("\n\n=== STEPS ===\n")

        for step in stats["report"]:
            f.write(
                f"step={step['step']} phase={step['phase']} reward={step.get('reward')} "
                f"done={step.get('done')}\n"
            )

            if "player" in step:
                f.write(f"  player={step['player']}\n")
            if "legal_actions_before" in step:
                f.write(f"  legal_actions_before={step['legal_actions_before']}\n")
            if "env_action" in step:
                f.write(f"  env_action={step['env_action']}\n")
            if "action_dict" in step:
                f.write(f"  action_dict={step['action_dict']}\n")
            if "roll" in step:
                f.write(f"  roll={step['roll']}\n")
            if "setup_phase" in step:
                f.write(f"  setup_phase={step['setup_phase']}\n")
            if "setup_stage" in step:
                f.write(f"  setup_stage={step['setup_stage']}\n")
            if "robber_event" in step and step["robber_event"] is not None:
                f.write(f"  robber_event={step['robber_event']}\n")
            if "value" in step:
                f.write(f"  value={step['value']}\n")
            if "vp_before" in step:
                f.write(f"  vp_before={step['vp_before']}\n")
            if "vp_after" in step:
                f.write(f"  vp_after={step['vp_after']}\n")
            if "resources_before" in step:
                f.write(f"  resources_before={step['resources_before']}\n")
            if "resources_after" in step:
                f.write(f"  resources_after={step['resources_after']}\n")
            if "resource_delta" in step:
                f.write(f"  resource_delta={step['resource_delta']}\n")
            if "tom" in step and step["tom"] is not None:
                f.write(f"  tom={step['tom']}\n")
            f.write("\n")

    print(f"Saved detailed game log to {out_path}")


if __name__ == "__main__":
    main()