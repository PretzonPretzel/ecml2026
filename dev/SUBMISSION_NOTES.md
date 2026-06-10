# Submission Notes (internal)

## Architecture

`submission.my_policy.MyPolicy` = `PlannerPolicy` ([or_policy.py](../submission/or_policy.py)):
a global SIPP-based dispatcher. The observation builder hands `(env, handle)`
to the policy; the policy runs `PlannerRuntime.act(env)` once per step.

- [netgraph.py](../submission/planner/netgraph.py): directed (cell, dir) graph
  from rail transitions; resources (level-free crossings split per axis);
  single-track corridor segments with per-node traversal sense; BFS
  heuristic tables per goal set.
- [sipp.py](../submission/planner/sipp.py): Safe-Interval Path Planning per
  train over a shared reservation table. Exact flatland 4.2.6 timing
  (fractional speeds via SpeedCounter recurrence, placement semantics,
  swap protection via vacate_to). Cost = time + 250·v per halt event.
  Two structural mechanisms:
  * **overstay rights**: a replanning train may always keep its current
    cell; followers' stale claims are woken (wake-ALL beats wake-earliest).
  * **directional corridor locks** (`LOCK_MIN_LEN=20`): plans traversing
    long single-track arteries lock them for direction+window; opposing
    plans schedule after; stubs don't lock. Prevents face-off freezes.
- [runtime.py](../submission/planner/runtime.py): plans all trains by slack
  priority at t=0; per step detects deviations, replans dirty agents
  (full SIPP, budgeted, most-urgent first), emits actions. Safety layers:
  reservation owner-check at cell entry (opposing claimants), corridor
  guard vs hopeless trains, provable mutual-pair deadlock containment,
  honest stub blocks (planless trains block their cell to horizon),
  runtime governor (degrades effort past PLANNER_TIME_BUDGET_S=1200s).

## Strategy facts (flatland 4.2.6 == evaluator version)

- Braking = collision penalty (−250·v): minimize halt events; hold trains
  off-map (free) instead of stopping on-map.
- Intermediate stops are SKIPPED deliberately (−50 < −250·v serve cost);
  `PLANNER_SERVE_STOPS=1` re-enables opportunistic serving (kept off: it
  triggers departure-delay malfunctions at stations).
- Cancellation = −5·SP-time ⇒ always launch trains.
- If flatland ever fixes the braking penalty (4.2.7+), revisit serve-stops!

## Results

Competition (sipp-v3 = v3-sipp-slim image, 2026-06-10, submission
f15cd299): **20.74/35 normalized, 72.9% complete** — leaderboard rank-1
score at that time was 12.49. Per level (norm of 5 / %done):
L0 3.96/100, L1 4.44/100, L2 4.20/100, L3 3.14/78, L4 1.67/56,
L5 2.38/61, L6 0.94/16. v4 (corridor locks) unsubmitted — daily cap.

Local suites (v4 config):

| suite | score | %done |
|---|---|---|
| example curriculum (27 envs, max 27) | 25.35 | 100.0% |
| official-generator level-1/2-like | 0.879 / 0.891 | 100% |
| 532 agents ll-8 mixed speeds, no malf | 0.971 | 100% |
| storm stress 62 / 118 agents + heavy malf | 0.521 / 0.803 | 81% / 93% |

RL baseline (shipped checkpoint): curriculum 15.55, 45.6% done.

## Env-var knobs (defaults = measured best)

PLANNER_SERVE_STOPS=0, PLANNER_CORRIDOR_GUARD=1, PLANNER_OWNER_CHECK=1,
PLANNER_EAGER_REPAIR=0, PLANNER_SHIFT_REPAIR=0, PLANNER_ESCAPE_MOVES=0,
PLANNER_FLAT_BACKOFF=0, PLANNER_DEADLOCK_EVERY=5, PLANNER_EXP_BUDGET=300000,
PLANNER_LOCK_MIN_LEN=20, PLANNER_HEADWAY=0, PLANNER_TIME_BUDGET_S=1200.
Tested-and-rejected (see memory/planner-progress-log): headway≥1,
cascade replanning, relaxed fallback, shift/eager repair, escape moves,
stub corridor locks, LOCK_MIN_LEN ∈ {all, 12}.

## Ship flow

1. Commit → `git push solution main` (solution = darshanmakwana412/ecml2026).
2. `gh workflow run docker.yml --repo darshanmakwana412/ecml2026 -f tag=<vN>`
   → image `ghcr.io/darshanmakwana412/ecml2026:<vN>` (amd64-only, slim
   ~825MB; the 7.3GB baselines base timed out the evaluator's 20-min pull
   window — keep images slim!).
3. `Checks` workflow = free E2E (docker build + full scenario run).
4. Portal: competition.flatland.cloud → new submission → image URL,
   repo URL, track **non-RL**. **Daily cap 2 — failed submissions count.**
5. Track progress: `curl api-competition.flatland.cloud/submissions/<id>/statuses`
   (public, no auth). Results: `/results/submissions/<id>` and
   `/results/submissions/<id>/tests/<test_ids>`. Leaderboard:
   `/results/benchmarks/c85d5fc2-15da-4a62-8e14-28d1261c29bd`.
