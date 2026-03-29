from __future__ import annotations

import argparse
from typing import Any, Dict

import torch

from environment.catan_env import CatanEnv
from learning.dqn.dqn_trainer import DQNTrainer
from learning.hybrid.hybrid_checkpoint import HybridCheckpointManager


def run_episode(
    env: CatanEnv,
    dqn_trainer: DQNTrainer,
    trade_policy,
    device: str = "cpu",
) -> Dict[str, Any]:
    obs = env.reset()
    done = False

    total_reward = 0.0
    steps = 0

    trade_counts = {
        "propose": 0,
        "accept": 0,
        "reject": 0,
        "counter": 0,
        "skip": 0,
    }

    while not done:
        controller = env.get_controller()

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if controller == "gameplay":
            action_mask = torch.tensor(
                env.get_action_mask(),
                dtype=torch.float32,
                device=device,
            )

            action = dqn_trainer.act(
                obs_tensor,
                action_mask,
                epsilon=0.0,
            )

            next_obs, reward, done, info = env.step(action)

        elif controller == "trade":
            with torch.no_grad():
                value, action_dict, log_prob, _, tom = trade_policy.act(
                    obs_tensor,
                    deterministic=True,
                )

            next_obs, reward, done, info = env.step(action_dict)

            action_type = action_dict.get("type", None)
            if action_type in trade_counts:
                trade_counts[action_type] += 1

        else:
            next_obs, reward, done, info = env.step(None)

        total_reward += reward
        steps += 1

        if done:
            break

        obs = next_obs

    result = {
        "total_reward": total_reward,
        "steps": steps,
        "trade_counts": trade_counts,
    }

    if isinstance(info, dict):
        if "winner" in info:
            result["winner"] = info["winner"]

    return result


def evaluate(
    num_episodes: int,
    env_config: Dict[str, Any],
    dqn_trainer: DQNTrainer,
    trade_policy,
    device: str,
) -> None:
    results = []

    for _ in range(num_episodes):
        env = CatanEnv(**env_config)
        result = run_episode(env, dqn_trainer, trade_policy, device=device)
        results.append(result)

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    total_trades = {
        "propose": 0,
        "accept": 0,
        "reject": 0,
        "counter": 0,
        "skip": 0,
    }

    for r in results:
        for k in total_trades:
            total_trades[k] += r["trade_counts"][k]

    print("Evaluation Results")
    print(f"Episodes: {num_episodes}")
    print(f"Avg Reward: {avg_reward:.4f}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print("Trade Counts:", total_trades)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = args.device

    state_dim = 256
    num_actions = 128

    dqn_trainer = DQNTrainer(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
    )

    trade_policy = None

    checkpoint_manager = HybridCheckpointManager()
    checkpoint = checkpoint_manager.load(
        args.checkpoint,
        dqn_trainer,
        trade_trainer=type("Dummy", (), {"policy": trade_policy, "optimizer": None})(),
        map_location=device,
    )

    print("Loaded checkpoint:", args.checkpoint)

    env_config = {}

    evaluate(
        num_episodes=args.episodes,
        env_config=env_config,
        dqn_trainer=dqn_trainer,
        trade_policy=trade_policy,
        device=device,
    )


if __name__ == "__main__":
    main()