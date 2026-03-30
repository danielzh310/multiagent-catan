import argparse
import os
import torch

from learning.unified.unified_policy import UnifiedPolicy
from learning.unified.unified_rollout_manager import UnifiedRolloutManager


def parse_args():
    parser = argparse.ArgumentParser(description="Play one game from a unified checkpoint and print actions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy deterministic actions")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation steps")
    parser.add_argument("--show-obs", action="store_true", help="Print raw observation vectors each step")
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
            if phase.name in ("MAIN_ACTION", "END_TURN")
            else "trade"
            if phase.name in ("TRADE_PROPOSE", "TRADE_RESPOND")
            else "auto"
        )

        resources_before = snapshot_resources(env)
        vp_before = snapshot_vp(env)

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
                    "roll": last_roll,
                    "robber_event": robber_event,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "vp": vp_after,
                    "resources_before": resources_before,
                    "resources_after": resources_after,
                    "resource_delta": resource_delta,
                }
            )
            total_steps += 1
            continue

        current_player = env.get_current_player_id()
        obs = env_manager._build_obs(env)

        if show_obs:
            print("obs=", obs)

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
            env_action = env_manager._decode_trade(action_dict, env)

        next_obs, reward, done, info = env.step(env_action)

        last_roll = next_obs["game"].get("last_roll")
        robber_event = next_obs["game"].get("last_robber_event")

        resources_after = snapshot_resources(env)
        vp_after = snapshot_vp(env)
        resource_delta = diff_resources(resources_before, resources_after)

        print(f"step={total_steps} phase={phase.name} player={current_player}")
        print(f" env_action={env_action}")
        print(
            " action_dict={"
            + ", ".join(f"{k}:{v.detach().cpu().numpy().tolist()}" for k, v in action_dict.items())
            + "}"
        )
        print(f" roll={last_roll}")
        if robber_event is not None:
            print(f" robber_event={robber_event}")
        print(f" reward={reward:.3f} done={done}")
        print(f" vp={vp_after}")
        if resource_delta:
            print(f" resource_delta={resource_delta}")
        print(f" resources={resources_after}\n")

        report.append(
            {
                "step": total_steps,
                "phase": phase.name,
                "player": str(current_player),
                "action_dict": {k: v.detach().cpu().numpy().tolist() for k, v in action_dict.items()},
                "env_action": env_action,
                "roll": last_roll,
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
        gameplay_only=args.gameplay_only,
    )

    print("\n=== GAME RESULT ===")
    print(f"winner={stats['winner']}")
    print(f"victory_points={stats['victory_points']}")
    print(f"total_steps={stats['total_steps']}")
    print(f"done={stats['done']}")

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
        f.write("=== STEPS ===\n")

        for step in stats["report"]:
            f.write(
                f"step={step['step']} phase={step['phase']} reward={step.get('reward')} "
                f"done={step.get('done')}\n"
            )

            if "player" in step:
                f.write(f"  player={step['player']}\n")
            if "env_action" in step:
                f.write(f"  env_action={step['env_action']}\n")
            if "action_dict" in step:
                f.write(f"  action_dict={step['action_dict']}\n")
            if "roll" in step:
                f.write(f"  roll={step['roll']}\n")
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