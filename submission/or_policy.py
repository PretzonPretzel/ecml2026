"""
Policy + observation builder pair that bridges the global SIPP planner into
the flatland policy interface. The observation for each agent is simply
(env, handle); the policy runs the planner once per step on the shared env.
"""
from typing import Any, Dict, List

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.RailEnvPolicy import RailEnvPolicy

from submission.planner.runtime import PlannerRuntime


class PlannerObsBuilder(ObservationBuilder):
    def reset(self):
        pass

    def get(self, handle: int = 0):
        return (self.env, handle)

    def get_many(self, handles=None):
        if handles is None:
            handles = range(self.env.get_num_agents())
        return {h: (self.env, h) for h in handles}


class PlannerPolicy(RailEnvPolicy):
    def __init__(self):
        super().__init__()
        self.runtime = PlannerRuntime()

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, int]:
        env = observations[0][0]
        return self.runtime.act(env)

    def act(self, observation: Any, **kwargs):
        env, handle = observation
        return self.runtime.act(env)[handle]
