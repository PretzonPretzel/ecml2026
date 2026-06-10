"""Quick scorer for the two malfunction-storm envs (used for A/B tests)."""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.step_utils.states import TrainState

from dev.eval_lib import ECML2026FineRewards
from submission.or_policy import PlannerObsBuilder, PlannerPolicy

ENVS = [
    "dev/big_envs/big_scene_1_a-62_ll-4.pkl",
    "dev/big_envs/big_scene_4_a-118_ll-6.pkl",
]

total = 0.0
for path in ENVS:
    env, _ = RailEnvPersister.load_new(path, obs_builder=PlannerObsBuilder(), rewards=ECML2026FineRewards())
    policy = PlannerPolicy()
    obs = env._get_observations()
    cum = {h: env.rewards.empty() for h in env.get_agent_handles()}
    halts = 0
    prev = [a.state for a in env.agents]
    t0 = time.perf_counter()
    for t in range(env._max_episode_steps):
        actions = policy.act_many(env.get_agent_handles(), list(obs.values()))
        obs, rew, dones, infos = env.step(actions)
        for h in env.get_agent_handles():
            cum[h] = env.rewards.cumulate(cum[h], rew[h])
            if prev[h] == TrainState.MOVING and env.agents[h].state == TrainState.STOPPED:
                halts += 1
            prev[h] = env.agents[h].state
        if dones["__all__"]:
            break
    n, ms = env.get_num_agents(), env._max_episode_steps
    norm = sum(max(sum(cum[h].values()), -ms) for h in range(n)) / (ms * n) + 1
    done = sum(1 for a in env.agents if a.state == TrainState.DONE)
    total += norm
    print(f"{path.split('/')[-1]:34s} norm={norm:.4f} done={done}/{n} halts={halts} "
          f"dead={len(policy.runtime._dead)} wall={time.perf_counter()-t0:.0f}s")
print(f"TOTAL {total:.4f}")
