"""
Local evaluation harness mirroring the ECML2026 server-side evaluation.

Loads scenario pkls with ECML2026 reward parametrization (fine-grained so we
get per-penalty breakdowns), runs a policy + obs builder exactly like
flatland-trajectory-generate-from-policy does, and reports the official
normalized reward plus diagnostics.
"""
import importlib
import time
from typing import Any, Dict, Optional

import numpy as np
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import BaseDefaultRewards
from flatland.envs.step_utils.states import TrainState


class ECML2026FineRewards(BaseDefaultRewards):
    """Same factors as flatland.envs.rewards.ECML2026Rewards, but dict-valued
    so we can break the score down by penalty type."""

    def __init__(self):
        super().__init__(
            cancellation_factor=5.0,
            cancellation_time_buffer=0.0,
            target_not_reached_minimum_penalty=100.0,
            intermediate_not_served_penalty=50.0,
            intermediate_late_arrival_penalty_factor=0.5,
            intermediate_early_departure_penalty_factor=0.5,
            collision_factor=250.0,
        )


def resolve(spec: str):
    """'pkg.mod.ClassName' -> class"""
    mod, _, cls = spec.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


def run_episode(
    env_path: str,
    policy_spec: str = "submission.my_policy.MyPolicy",
    obs_builder_spec: str = "submission.my_observation_builder.MyObservationBuilder",
    max_steps_override: Optional[int] = None,
    verbose: bool = False,
    no_malfunction: bool = False,
) -> Dict[str, Any]:
    policy = resolve(policy_spec)()
    obs_builder = resolve(obs_builder_spec)()
    rewards = ECML2026FineRewards()

    t0 = time.perf_counter()
    env, _ = RailEnvPersister.load_new(env_path, obs_builder=obs_builder, rewards=rewards)
    if no_malfunction:
        from flatland.core.effects_generator import EffectsGenerator
        env.effects_generator = EffectsGenerator()  # no-op: levels 0-2 have none
    load_time = time.perf_counter() - t0

    n_agents = env.get_num_agents()
    max_steps = env._max_episode_steps
    observations = env._get_observations()

    cum = {h: rewards.empty() for h in range(n_agents)}
    halts = 0  # MOVING->STOPPED transitions (the -250*v events)
    prev_states = [a.state for a in env.agents]

    t0 = time.perf_counter()
    end_step = max_steps if max_steps_override is None else min(max_steps, max_steps_override)
    steps = 0
    for _ in range(end_step):
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        observations, step_rewards, dones, infos = env.step(action_dict)
        steps += 1
        for h in range(n_agents):
            cum[h] = rewards.cumulate(cum[h], step_rewards[h])
            st = env.agents[h].state
            if prev_states[h] == TrainState.MOVING and st == TrainState.STOPPED:
                halts += 1
            prev_states[h] = st
        if dones["__all__"]:
            break
    run_time = time.perf_counter() - t0

    # official normalization (per-agent clip at -max_episode_steps)
    per_agent_total = [sum(cum[h].values()) for h in range(n_agents)]
    normalized = float(
        sum(max(r, -max_steps) for r in per_agent_total) / (max_steps * n_agents) + 1
    )
    pct_done = float(sum(a.state == TrainState.DONE for a in env.agents)) / n_agents

    breakdown = {k: float(sum(cum[h][k] for h in range(n_agents))) for k in cum[0]}

    result = dict(
        env=env_path.split("/")[-1].replace(".pkl", ""),
        n_agents=n_agents,
        max_steps=max_steps,
        steps=steps,
        normalized_reward=round(normalized, 4),
        pct_done=round(pct_done, 4),
        halts=halts,
        load_s=round(load_time, 1),
        run_s=round(run_time, 1),
        **{k.lower(): round(v, 1) for k, v in breakdown.items()},
    )
    if verbose:
        print(result)
    return result


def _worker(args):
    try:
        return run_episode(*args)
    except Exception as e:  # surface failures as rows, don't kill the pool
        import traceback
        return dict(env=args[0].split("/")[-1], error=f"{e.__class__.__name__}: {e}",
                    trace=traceback.format_exc()[-1500:])
