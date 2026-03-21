from dataclasses import dataclass


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5

    gamma: float = 0.999
    gae_lambda: float = 0.95

    clip_param: float = 0.2
    ppo_epochs: int = 10
    num_minibatches: int = 64
    value_loss_coef: float = 1.0
    max_grad_norm: float = 0.5

    entropy_coef_start: float = 0.04
    entropy_coef_final: float = 0.005
    entropy_anneal_start: int = 500
    entropy_anneal_end: int = 1500

    num_processes: int = 2
    num_envs_per_process: int = 2
    num_steps: int = 32
    truncated_seq_len: int = 8

    total_env_steps: int = 4096

    eval_every: int = 5
    num_eval_episodes: int = 8
    num_eval_processes: int = 2

    num_policies_to_store: int = 50
    add_policy_every: int = 4
    update_opponents_every: int = 1

    recompute_returns: bool = True
    normalize_values: bool = True

    seed: int = 0
    use_cuda: bool = False
    use_linear_lr_decay: bool = True
    experiment_id: str = "catan_debug"
    checkpoint_path: str = "checkpoints"
    load_from_checkpoint: bool = False