"""
Generate competition-scale test envs (level 1-6-like): many agents, longer
lines (up to 6 intermediate stops), mixed train speeds, on the official
competition topology from level_0_scenario_1.pkl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards
from reinforcement_learning.create_curriculum_envs import create_curriculum_env
from reinforcement_learning.sampling.sampling_env_generator import sampling_env_generator

OUT = Path("dev/big_envs")

# (scene, n_agents, line_length, speed_mix)
SPEEDS = {1.0: 0.4, 0.8: 0.2, 0.6: 0.2, 0.5: 0.2}
CONFIGS = [
    ("scene_1", 62, 4, None),
    ("scene_4", 118, 6, None),
    ("scene_5", 90, 4, SPEEDS),
    ("scene_5", 150, 6, SPEEDS),
    ("scene_5", 300, 8, SPEEDS),
    ("scene_5", 532, 8, SPEEDS),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for scene, n_agents, ll, speeds in CONFIGS:
        env = create_curriculum_env(scene=scene, n_agents_range=None, line_length=ll)
        env.number_of_agents = n_agents
        if speeds is not None:
            env.line_generator = sparse_line_generator(speed_ratio_map=speeds, line_length=ll)
        env.reset()
        name = f"big_{scene}_a-{n_agents}_ll-{ll}{'_mix' if speeds else ''}"
        out = OUT / f"{name}.pkl"
        RailEnvPersister.save(env, str(out))
        print(f"saved {out} agents={env.get_num_agents()} max_steps={env._max_episode_steps}")


if __name__ == "__main__":
    main()
