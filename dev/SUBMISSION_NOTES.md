# Submission Notes (internal)

## Architecture

`submission.my_policy.MyPolicy` = `PlannerPolicy` ([or_policy.py](../submission/or_policy.py)):
a global SIPP-based dispatcher. The observation builder hands `(env, handle)`
to the policy; the policy runs `PlannerRuntime.act(env)` once per step.

- [netgraph.py](../submission/planner/netgraph.py): directed (cell, dir) graph
  from rail transitions; resources (level-free crossings split per axis);
  single-track corridor segments; BFS heuristic tables per goal set.
- [sipp.py](../submission/planner/sipp.py): Safe-Interval Path Planning per
  train over a shared reservation table. Exact flatland 4.2.6 timing
  (fractional speeds via SpeedCounter recurrence, placement semantics,
  swap protection via vacate_to). Cost = time + 250·v per halt event
  (the env charges −250·v on every MOVING→STOPPED transition, even clean
  braking — verified empirically, see memory/flatland-426-mechanics).
- [runtime.py](../submission/planner/runtime.py): plans all trains by slack
  priority at t=0; per step detects deviations (malfunctions etc.),
  replans dirty agents (full SIPP, budgeted, most-urgent first), and emits
  actions. Safety layers: reservation owner-check at cell entry
  (opposing-direction claimants only), corridor guard against hopeless
  (dead/planless) trains, provable mutual-pair deadlock detection with
  permanent blocks.

## Strategy facts (flatland 4.2.6 == evaluator version)

- Braking = collision penalty (−250·v): minimize halt events; hold trains
  off-map (free) instead of stopping on-map.
- Intermediate stops are SKIPPED deliberately (−50 < −250·v serve cost);
  `PLANNER_SERVE_STOPS=1` re-enables opportunistic serving (off: it triggers
  departure-delay malfunctions at stations and destabilizes dense scenarios).
- Cancellation = −5·SP-time ⇒ always launch trains.
- If flatland ever fixes the braking penalty (4.2.7+), revisit serve-stops!

## Local results (2026-06-10)

| suite | score | %done |
|---|---|---|
| example curriculum (27 envs, max 27) | 25.13 | 98.8% |
| official-generator level-1/2-like | 0.879 / 0.891 | 100% |
| 532 agents ll-8 mixed speeds, no malf | 0.96 | 97% |
| storm stress (62/118 agents + heavy malfunctions) | 0.61 / 0.50 | 100% / 35% |

RL baseline (shipped checkpoint) for reference: curriculum 15.55, 45.6% done.

## Env-var knobs (defaults are the measured best)

PLANNER_SERVE_STOPS=0, PLANNER_CORRIDOR_GUARD=1, PLANNER_OWNER_CHECK=1,
PLANNER_EAGER_REPAIR=0, PLANNER_SHIFT_REPAIR=0, PLANNER_FLAT_BACKOFF=0,
PLANNER_DEADLOCK_EVERY=5, PLANNER_EXP_BUDGET=300000, PLANNER_HEADWAY=0.

## Submission flow

1. Push this repo to the team's private GitHub repo.
2. Run GH action `docker` (Actions tab → "Build ECML 2026 Starterkit Docker
   Image" → choose a tag, e.g. `v1-sipp`).
3. Repo → Packages → package settings → invite **Flatland Competition**
   account with *read* access.
4. competition.flatland.cloud → new submission → paste image URL
   `ghcr.io/<org>/<repo>:<tag>`.
5. Limits: 2 submissions/day; 30 min/scenario; 4 CPU / 15 GB; no GPU.
