# Handoff: ECML2026 SIPP dispatcher, day 1

Date: 2026-06-10. Session: competition kickoff to first scored submission.

## What was accomplished

- Understood the full competition: ECML 2026 Flatland Real-World Baselines
  Challenge, 7 levels x 5 scenarios (files level_0..level_6, max score 35),
  deadline 2026-06-29 AoE, two tracks (we are non-RL/OR), daily cap 2
  submissions (failed ones count), eval limits 4 CPU / 15 GB / 30 min per
  scenario / approx 4h total / 20 min pod start window.
- Built the solution: a global SIPP-based dispatcher under
  `submission/planner/` (netgraph.py, sipp.py, runtime.py) wired through
  `submission/or_policy.py`. The observation builder passes (env, handle);
  the policy plans all trains over a shared reservation table and emits
  per-step actions.
- Built local tooling under `dev/`: eval harness with the official scoring
  formula and per-penalty breakdowns (`dev/evaluate.py`, `dev/eval_lib.py`),
  storm quick-scorer (`dev/quick_storm.py`), competition-scale env
  generators (`dev/make_big_envs.py`, `dev/big_envs/`), level-1/2 replicas
  generated with the organizers' scenario generator (`dev/level_envs/`,
  requires the flatland-scenarios repo cloned as a sibling directory).
- Submission infrastructure: private repo darshanmakwana412/ecml2026
  (remote `solution`), GH action `docker.yml` builds amd64-only slim image
  (python:3.12-slim + flatland-rl, 825 MB), `Checks` workflow runs a full
  scenario through the image as E2E. Ship flow documented in
  `dev/SUBMISSION_NOTES.md`.
- Submitted and scored: sipp-v3 (image v3-sipp-slim) scored
  **20.74/35 normalized, 72.9 percent complete**. Public leaderboard
  rank 1 at that moment was 12.49, so this should take the lead once
  listed. Per level (norm of 5 / pct done): L0 3.96/100, L1 4.44/100,
  L2 4.20/100, L3 3.14/78, L4 1.67/56, L5 2.38/61, L6 0.94/16.
- v4 (image `v4-sipp-locks`, commit 4d11acc) is built, CI-green, and
  NOT yet submitted (daily cap). It adds directional corridor locks,
  the biggest robustness win of the day.

## Key decisions

- Strategy is shaped by a verified flatland 4.2.6 quirk: ANY
  MOVING->STOPPED transition costs 250 x speed (clean braking included,
  AgentTransitionData.speed is captured pre-action). Therefore: minimize
  halt events, hold trains off-map for timing (free), and deliberately
  SKIP intermediate stops (-50 each) because serving requires a stop
  (-250 x v, v >= 0.5 in this competition). If a flatland release ever
  fixes this, flip PLANNER_SERVE_STOPS and revisit.
- Overstay rights (sipp.py): a replanning train may always keep the cell
  it physically occupies; followers' stale reservation claims are woken
  (wake ALL claimants of the start visit, wake-earliest tested worse).
  This unfroze 70-train queues behind breakdowns.
- Directional corridor locks (sipp.py, LOCK_MIN_LEN=20): plans traversing
  single-track corridors of >= 20 cells lock direction+window; opposing
  plans schedule after. Locking ALL corridors or >= 12 over-serializes
  stations and tested worse. Stub blocks must NOT lock (tested worse both
  on day-1 attempts).
- Tested and rejected (do not relitigate, measurements in
  memory/planner-progress-log and dev/results/*.csv): headway >= 1
  (catastrophic with the braking penalty), cascade replanning, relaxed
  fallback planning, shift-repair and eager-repair (preserve doomed
  routes; full SIPP replans reroute better), escape moves (net negative,
  env-gated off), 40k expansion budget (300k better).
- Slim Docker image is mandatory: the official 7.3 GB baselines base
  image timed out the evaluator's 20 min pod-start window (first
  submission failed exactly at 1200s Pending). Keep images small.
- Honest stub blocks: a plan-less on-map train blocks its cell to
  horizon until a replan succeeds, otherwise others plan waits next to
  short stubs and wedge.

## Current state

- Branch main, all work committed and pushed to both the local clone and
  the `solution` remote (darshanmakwana412/ecml2026). Last commit
  5f8578d (cleanup), v4 image built from 4d11acc, planner code identical
  between them.
- Local bests with v4 config: example curriculum 25.35/27 at 100.0
  percent delivery; storm stress 1.3244 total (scene_1 50/62, scene_4
  110/118); level-like 0.879/0.891 at 100 percent; 532-agent no-malf
  0.971 at 100 percent.
- Tasks: submission pipeline proven end to end; storm robustness much
  improved but scene_1 retains a 12-train residual face-off; level_6 is
  the big open problem.

## Next session plan

1. Submit v4 first thing (daily cap resets): image
   `ghcr.io/darshanmakwana412/ecml2026:v4-sipp-locks`, repo URL
   https://github.com/darshanmakwana412/ecml2026, track non-RL. The user
   (Darshan) pastes it in the portal. Expect gains on L3-L5 (locally the
   locks took the storm proxy from 37 to 93 percent done).
2. Level_6 forensics (0.94/5, 15.6 percent done, the largest sitting
   opportunity at roughly +4 points). "Infrastructure disruptions" is
   likely ConditionalMalfunctionEffectsGenerator with
   condition_stopped_cells_and_range (malfunctions triggered when trains
   are STOPPED on specific cells) plus heavier breakdown params, since
   that is the only machinery in flatland 4.2.6 matching the levelconfig
   description. Build a local level-6 replica (532 agents, scene_5,
   heavy malfunctions, stopped-cells conditions on key junctions) and
   measure. Candidate defenses: avoid halting on switch/junction cells
   (halt placement preference in SIPP), stronger jam containment.
3. Compare v3 vs v4 per-level deltas when v4 scores land; the API is
   public, endpoints in dev/SUBMISSION_NOTES.md, test ids in the
   planner-progress-log memory. Monitor pattern: poll
   /submissions/<id>/statuses every 180s.
4. If locks transfer well, the next structural ideas for storm levels:
   junction-area wedge prevention (the remaining scene_1 face-off is 12
   trains around a junction cluster), LNS-style reinsertion of the worst
   late trains, and serve-stops re-evaluation per level.
5. Consider the RL track later (separate prizes, needs evidence of
   training; imitation-learn the planner is the obvious route).

## Context a fresh agent needs

- Memory files (auto-loaded) hold the durable facts: competition rules
  (ecml2026-competition-facts), the 4.2.6 mechanics incl. the braking
  penalty (flatland-426-mechanics), version-by-version verdicts
  (planner-progress-log), infra and ship flow (submission-infra).
- Local python env: `.venv` (3.12, flatland-rl 4.2.6, torch). Eval:
  `.venv/bin/python -m dev.evaluate --glob '<pkls>' --policy
  submission.or_policy.PlannerPolicy --obs-builder
  submission.or_policy.PlannerObsBuilder --jobs 9`.
- The curriculum env pkls live in
  reinforcement_learning/curriculum/curriculum/example_curriculum/ after
  unzipping example_curriculum.zip (gitignored, re-unzip if missing).
- The submission id of sipp-v3 is f15cd299-d977-4068-ab04-b9c237625f20;
  test ids for levels 0..6 are in the statuses endpoint output and in
  the planner-progress-log memory entry.
- Docker on this Mac needs DOCKER_CONFIG=/tmp/docker-clean-config for
  local builds (credsStore desktop is broken in sandbox); CI builds are
  the source of truth anyway.
