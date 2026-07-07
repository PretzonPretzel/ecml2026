# Winning ECML 2026 Flatland with a SIPP Train Dispatcher

This is a consolidated writeup of my [ECML 2026 Flatland Real-World Baselines Challenge solution](https://competition.flatland.cloud) that achieved rank 1 on the leaderboard. This writeup consolidates all the experimentations and implementation details from my obsidian vault, claude has been used to write parts of this writeup and structure the content, experiments and findings

## The Flatland Competition

Flatland is a train scheduling problem where we are given a railway, a fleet of trains with timetables, and we have to route every train to its destination without them crashing into each other. The competition fixes one railway topology, a 120x150 grid with 28 stations, and has 7 difficulty levels, 5 scenarios each, for 35 scored episodes. Each scenario spawns anywhere from 8 to 532 trains onto that same map. Here is what one scenario with 25 trains looks like in Flatland's own renderer, cartoon trains and grass and all:

| | |
| --- | --- |
| ![The same scenario in Flatland's own renderer, cartoon trains and grass](assets/flatland_hero_stock.png) | ![The same scenario redrawn by flatviz as the graph the solver reasons over](assets/flatland_hero.png) | 

Every train starts on the map at some origin and has to reach a destination, which are the red stars, and the grey diamonds are switches. The dense knot of switches in the middle is a station throat, where a lot of routes funnel together and most of the interesting conflicts happen. Each cell has a set of allowed entry to exit direction transitions, which means a train entering a cell heading north can only leave in the directions that cell permits for a northbound entry, so most cells have exactly one exit, but a switch has a real routing choice:

![A switch is a routing choice: each (cell, heading) is a graph node](assets/flatland_junction.png)

A train at cell `(r, c)` heading east is a different search node from the same train heading west, because they can do different things next. We encode a node as `((r * W) + c) * 4 + d` and the whole planner lives on this directed graph of `4 * H * W` nodes, and a switch is just a node where some entry direction has more than one successor.

On top of the geometry sits a timetable. Each train has an earliest departure, a latest arrival, and a list of intermediate stops it is supposed to serve on the way, so think of an intercity that calls at three platforms before its terminus:

![One train's timetable: origin, intermediate stops, destination](assets/flatland_timetable.png)

Trains also move at different speeds, and Flatland speeds are fractions, so an intercity runs at `v = 1.0` which is one cell per step, a regional at `0.8`, a slow stopping service at `0.6`, and a commuter at `0.5`. A train at speed `p/q` banks `p/q` of distance every step and only leaves its cell once the banked distance reaches 1, so a `0.8` train needs two steps to leave its first cell, and because the leftover fraction carries over, some of its later cells clear in one step. The planner has to track this fraction exactly, because it needs to predict the precise step at which each cell frees up.

### Scoring

The score is where the whole game is decided. Per scenario every agent piles up penalties (all rewards are `<= 0`), and the normalized reward is

```
normalized = sum_over_agents( max(cum_reward_agent, -max_episode_steps) ) / (max_episode_steps * n_agents) + 1
```

so a perfect run scores 1.0 and a total disaster scores 0, and the competition score is just the sum of these over all 35 scenarios, so the max is 35. There is one hard rule on top of this, which is that evaluation aborts a whole level if fewer than 25% of trains finish it, so getting most trains home is a constraint and not just something nice to have.

The malfunctions ramp up with the levels, so levels 0 to 2 are clean, level 3 adds random breakdowns where a train freezes in place for some steps, levels 4 and 5 add random departure delays, and level 6 adds infrastructure disruptions on top of everything else. One important detail is that the evaluator does not run your policy live and trust it. Your container records a trajectory of actions with `flatland-trajectory-generate-from-policy`, and then a server-side `TrajectoryEvaluator` **replays those actions** through its own Flatland install to compute the score, so only the recorded actions matter. We ran our own trajectories through that exact code path so that there would be no surprise between the local score and the real one.

The RL baseline they ship, which is a FastTreeObs observation fed into an MLP policy trained on the example curriculum, scores **15.55 out of 27 on that curriculum and gets 45.6% of trains home**. That is the number to beat, and it is also the reason the organizers run a separate OR / non-RL track, because planners win Flatland, they always have, and the organizers know it.

## The reward function, and the one number that matters

Before writing a single line of the planner we sat down and read `flatland.envs.rewards` for the exact competition parameters, because in a problem like this the cost function is the strategy. Here is the full reward book:

![The ECML2026 reward book and the one number that dominates it](assets/flatland_rewardbook.png)

Most of these are what you would expect. Canceling a train (never departing it) costs `5x` its shortest-path time, which is large, so you always launch every train. Arriving late costs `1` per step. Skipping an intermediate stop costs a flat `50`. But look at the top bar, because a **collision costs `250 x speed`**, and here is the thing that took us a couple of hours of poking at the environment to actually believe:

> In Flatland 4.2.6, which is the version the evaluator runs, *any* transition from MOVING to STOPPED is charged the full collision penalty. Not just crashes. A clean, deliberate, perfectly safe brake of a moving train costs `250 x v`.

We checked this by hand instead of trusting the docstring (which says otherwise). Load a scenario, drive one intercity forward on empty track, brake it cleanly with nothing in front of it, and read the reward:

```python
# one speed-1 train, clean voluntary brake, empty track ahead
env, _ = RailEnvPersister.load_new(path, rewards=ECML2026Rewards())
# ... step it forward a few cells, then issue STOP_MOVING ...
# step_reward for that transition:  -250.0
```

It is `-250.0`, exactly. If you dig into `rail_env.py` the reason is that `AgentTransitionData.speed` is captured in the pre-step loop *before* the speed update, and the reward code multiplies the collision factor by that pre-step speed on every MOVING to STOPPED transition, whether it came from the motion check blocking you or from your own STOP action. Reaching the final target is a MOVING to DONE transition and is exempt, so arriving is free, but stopping short is not.

This one detail decides everything, and three things fall straight out of it.

**We skip every intermediate stop, on purpose.** Serving a stop means the train has to actually halt at the platform, and a halt is a MOVING to STOPPED transition, so it costs `250 x v`, while skipping the stop costs a flat `50`. So serving only wins when `250 x v < 50`, which is `v < 0.2`, and every real train in this competition runs at `0.5` or faster:

![Serve vs skip break-even by train speed](assets/flatland_skipstops.png)

The break-even is at `v = 0.2` and every train category (commuter, stopping service, regional, intercity) sits above it, so the right move is to blow through every platform without stopping and just eat the `-50`. It feels wrong because an intercity is supposed to stop at its platforms, but the reward book says skip, so we skip. We left one environment flag, `PLANNER_SERVE_STOPS`, that flips this back, in case the organizers ever fix the braking bug during the competition, but they did not.

**We hold trains off the map instead of stopping them on it.** A train that has not departed yet sits off the board for free with no penalty for waiting, but a train on the board that has to give way has to brake, which is `250 x v`. And delaying a departure by `D` steps only costs about `D` at the end, because late arrival is `-1` per step, so holding a train back is a slow linear cost while braking a moving one is a flat and expensive cost:

![Holding a train off-map is free; braking it on-map is not](assets/flatland_holdoffmap.png)

The two lines cross at `D = 250`, so you can hold an intercity off the map for 250 steps and still come out ahead of letting it brake even once. On a map where episodes only run a few hundred steps, that basically means never let a moving train stop if you can delay its departure instead.

**Minimize halts, full stop.** Every MOVING to STOPPED is `-250 x v`, so the whole job of the planner, on top of getting trains home, is to get them home without ever touching the brakes. This changes what we are even searching for, because we are no longer looking for shortest paths, we are looking for paths that never have to stop, and this is exactly the problem that Safe-Interval Path Planning was made for.

## V1: SIPP over a reservation table

The standard way to schedule a lot of agents on shared track is prioritized planning with a reservation table. You order the trains and plan them one at a time, so each train plans around the reservations of all the trains planned before it and then writes its own path into the table. It is not globally optimal like a full Conflict-Based Search would be, but it scales to hundreds of agents and with a good priority order it gets very close.

The core of it is [Safe-Interval Path Planning](https://www.cs.cmu.edu/~maxim/files/sipp_icra11.pdf). The idea behind SIPP is that you should not search over `(cell, timestep)` because there are far too many timesteps, you should search over `(cell, safe interval)`, where a safe interval is a maximal window during which a cell is free. A reservation table is really a 2D object, resource on one axis and time on the other, where every occupancy is a rectangle, so the gaps in a resource's column are its safe intervals, and a train's plan is a staircase that climbs through those gaps:

![SIPP threads a monotone staircase through the safe intervals](assets/flatland_sipp_spacetime.png)

This picture is the whole algorithm. Our train wants to get from cell A to cell F, and another train (train 7) has reserved cell D for the window `[3, 9)`, so our train rolls through A and B and then has to wait, because it cannot enter D until D frees at `t=9`. SIPP holds it in cell C until the exact moment D opens and then threads through, and that wait is a single halt, marked with the X, which costs `250 x v`. The search is looking for the staircase that reaches the goal with the fewest halts, and after that with the earliest arrival.

The reservation table maps every resource to a sorted list of `(start, end, agent)` occupancies, and the core query hands back the free gaps:

```python
def safe_intervals(self, res, t_from):
    """Maximal free intervals of `res` intersecting [t_from, horizon)."""
    lst = self.intervals.get(res)
    if not lst:
        return [(t_from, self.horizon)]
    out, lo = [], t_from
    i = bisect_right(lst, (t_from, 1 << 60, 1 << 60)) - 1
    if i >= 0 and lst[i][1] > t_from:      # inside an occupied block
        lo = lst[i][1]
    i += 1
    for s, e, _ in lst[i:]:
        if e <= lo:
            continue
        if s > lo:
            out.append((lo, min(s, self.horizon)))   # a gap
        lo = max(lo, e)
        if lo >= self.horizon:
            return out
    if lo < self.horizon:
        out.append((lo, self.horizon))
    return out
```

A resource is a cell, with one exception, which is that the map has level-free crossings (bridges) where two trains can sit in the same cell on different axes, so those split into two resources by axis. The A\* search expands `(node, fraction, interval)` states, and the thing that makes it exact for Flatland is the fractional-speed arithmetic, because for a train at speed `p/q` sitting in a cell with banked fraction numerator `fr`, the number of steps to roll out of the cell without stopping is `ceil((q - fr) / p)`. So from an entry time `t` the non-stop exit is `t + ceil((q - fr) / p)`, and the cost of a state is `arrival_time - earliest_possible + halt_weight * n_halts`. The A\* heuristic is the exact non-stop travel time left to the goal, which we get from a backward BFS over the graph, and it is admissible because non-stop is the fastest a train can ever go.

The successor step is where the halt cost enters. For each successor we look at the safe intervals of that successor's resource, find the earliest time we could enter it, and check whether entering that late meant we had to wait in the current cell, because waiting is a halt:

```python
for v_node in g.successors(node):
    for s, e in self.res.safe_intervals(res_v, exit_nonstop):
        if s > free_til:                 # can't wait in u past when u itself is claimed
            break
        tau = max(exit_nonstop, s)       # earliest legal entry into v
        # ... swap protection and corridor locks go here ...
        halted = tau > exit_nonstop      # did we have to wait?  that's a halt
        step_cost = (tau - t) + (halt_weight if halted else 0.0)
        ...
```

`halt_weight` is `250 x v`, straight out of the reward book, so the search literally prices a wait at the same rate the game prices a brake, which means it will happily send a train the long way around or hold it back at its origin just to avoid paying that `250`. This is SIPP behaving as a scheduler that has fully understood the one number that matters.

The last piece is the order we plan trains in. We plan tightest-slack-first, where slack is `latest_arrival - earliest_departure - nonstop_travel_time`, so the trains with the least room to spare get first pick of the reservation table:

```python
def slack(agent):
    dist = g.dist_to_goals(goal_nodes[agent.handle])[start_node(agent)]
    travel = ceil(dist / speed)
    return (agent.latest_arrival) - agent.earliest_departure - travel
order = sorted(env.agents, key=slack)
```

We wired this into the competition's policy interface (an observation builder hands `(env, handle)` to the policy, which runs the planner once and then just reads out the actions), and ran it on the example curriculum. **V1 scored 24.38 out of 27, with 93.8% of trains delivered**, against the RL baseline's 15.55 and 45.6%. So on the first day, before any of the hard problems, the planner was already 57% better than the RL policy they ship, and that is really the whole OR-beats-RL story in one number.

## V2: making it survive execution

A plan is only good until the world stops matching it, and in Flatland the world stops matching it all the time, because trains break down, departures slip, and sometimes our own placement attempts get blocked. So the runtime is a plan-and-execute loop. On the first step it builds the graph and plans everyone, and on every step after that it compares each train's actual state against its plan, marks the ones that have drifted as dirty, and replans them (a full SIPP re-search, most urgent first, under a per-step budget), while everything else just reads its next action out of the schedule.

Turning a timed path into actions has one subtlety that is worth calling out, and it is the braking penalty again. When a train is already stopped, say a malfunction just ended, and its plan says to wait a bit longer before moving, we have to send `DO_NOTHING` and keep it stopped and then start it at the last moment, instead of letting it inch forward and brake again, because restarting a stopped train is free but braking a moving one is `-250`. The action builder is full of this kind of care.

The bigger job in V2 was making the thing safe when the timing goes against us. Four mechanisms went in:

**Swap protection.** Flatland does not let two trains cross on a single edge in one step, so they cannot swap places. If our schedule ever asks for a swap, the replay quietly turns it into a stop (and a `-250`, and maybe a deadlock). So the reservation table also keeps a `vacate_to` map, and when agent A leaves resource `u` into `v` at time `tau`, we record that, and any other train that tries to enter `u` from `v` at that same instant gets pushed back a step.

![Swap protection and the single-track deadlock it prevents](assets/flatland_swap.png)

This looks like a small corner case, but it is actually the biggest failure mode in the whole competition, because two trains meeting head on in a single-track corridor cannot pass each other and cannot swap, so they just sit there facing each other forever. We will come back to this later.

**Owner checks at cell entry.** Even with reservations, schedules go stale between replans, so at the moment a train is about to enter a new cell we double-check the reservation table, and if the slot is claimed by an opposing train (one heading the other way, which is a real head-on risk) we hold. Same-direction conflicts we let resolve as normal queueing, because queueing behind someone going your way is no worse than waiting.

**Honest stub blocks.** When a train genuinely cannot find a plan, because the table is too crowded for a moment, we do not just let it sit there invisibly, we block its current cell in the reservation table all the way to the horizon, so every other train routes around it instead of queueing up behind it waiting for a train that is not moving. Once the train replans, the stub is replaced by its real reservations.

**Provable-deadlock containment.** Some jams really are dead and it is important not to thrash trying to fix them, so we mark only the provably dead trains, which are mutual pairs where each train's only possible next cell is the other train's current cell. Those get marked dead and told to sit still. Bigger cycles we leave alone, because a ring of trains in Flatland can sometimes rotate itself free, and killing them off by mistake loses trains.

There is also a runtime governor, because each scenario has a 30-minute wall-clock budget on the evaluation cluster, so if planning time creeps past a threshold the planner starts cutting its own search effort (fewer replans per step, smaller expansion budget) to make sure it always finishes stepping the environment, since a slightly worse plan is much better than a scenario that times out at zero.

**V2 scored 25.43 out of 27 (99.4% delivered)** on the curriculum, and the little bit that is left is mostly the stop-skipping, which is by design. On the clean levels the planner was basically done.

## Shipping it to the evaluator

This part deserves its own section because it cost us our entire first submission, and it is the kind of thing nobody writes down.

You submit to Flatland as a Docker image, so the evaluator pulls your image, runs your recorded-trajectory command, and scores it. The starter kit's base image is `flatland-baselines`, which is **7.26 GB compressed** because it carries conda, torch, ray, ffmpeg and the whole ML stack, none of which our planner needs, but we did not think about it, built on top of the base image, and submitted anyway.

The submission failed, and not with a crash, with this:

```
FAILURE  Elapsed time 1200.47s exceeded start time limit 1200s. Status: Pending
```

The evaluation pod gets 20 minutes to reach Running, and that includes pulling the image, and a 7.3 GB pull over the cluster's registry link did not make it in time, so the pod sat Pending for exactly 1200 seconds and got killed. We had burned a submission (and we later confirmed the hard way that **failed submissions still count against the daily cap of 2**) on a problem that had nothing to do with our code.

The fix was a slim image, just `python:3.12-slim` plus `flatland-rl==4.2.6` and nothing else, because the planner is pure Python and numpy, which is **825 MB instead of 7.3 GB** and pulls in seconds. We kept the fat one around as `Dockerfile.baselines` just in case. There was also half an hour of fighting with GitHub Container Registry permissions, because giving the evaluator read access to a private package quietly turns off "inherit access from source repo", which then takes away the CI's write access so the build starts failing, and the two settings live in different panels so you need both. The lesson we took from this is simple: keep the image slim, and test the full pull-and-run path before you spend a submission on it.

## The first real result

With the slim image (carrying a planner version we will get to, with overstay rights already folded in) we finally got a scored submission back, and here is where the whole thing stood after one day:

![Per-level competition results for sipp-v3](assets/flatland_perlevel.png)

That is **20.74 out of 35, with 72.9% of trains delivered**, and the public leaderboard leader at that moment was 12.49, so we were at rank 1 by a 66% margin, and our best planner version was still sitting unsubmitted because of the daily cap.

The per-level breakdown is the interesting part though, because it tells you exactly where the points are and turns "make it better" into an actual list:

- **Levels 0 to 2 (no malfunctions): 12.60 out of 15, 100% delivered.** This is working exactly as designed, and the gap from a perfect 15 is almost entirely our stop-skipping, which is the correct trade.
- **Levels 3 to 5 (malfunctions): 7.19 out of 15, 56 to 78% delivered.** This is the weak spot, because when trains break down our schedules cascade into jams and we lose a big chunk of the fleet, and this is where the points are.
- **Level 6 (infrastructure disruptions): 0.94 out of 5, 15.6% delivered.** This is almost a total loss, and it is a different kind of problem from the others.

So the clean levels were done, and the malfunction levels were where the competition was actually going to be won, and we had a guess about why they were failing.

## The malfunction levels: two big fixes

To work on the malfunction problem locally we built some deliberately nasty stress environments, 62, 118 and up to 532 agents on the real map with random origin-destination pairs, mixed speeds and heavy malfunction rates, harsher than the real levels on purpose so that whatever survived them would be robust. The 118-agent storm was the one we used the most, and two fixes came out of it, and together they took that environment from **34% of trains delivered to 93%**:

![Version progression and the two structural storm fixes](assets/flatland_progression.png)

### Fix 1: overstay rights

The first bug was subtle and nasty. When a train breaks down it sits in its cell for the malfunction duration, and every train queued behind it has to replan, but those followers had already reserved cells ahead of them, including the broken train's own cell, for times in the near future. So when the broken train tried to replan its own escape, SIPP saw those followers' reservations of its current cell as hard "you have to be gone by then" deadlines, and those were impossible to satisfy because the train was physically sitting there and could not leave until the malfunction ended, so SIPP returned no plan. Then the train went plan-less, which stubbed its cell, which made the followers replan, which re-reserved, and the whole queue behind one breakdown froze solid.

The fix we called **overstay rights**, which just says that a train that is physically in a cell may always stay in it, full stop. Its presence is the truth, and the other trains' reservations of that cell are stale fiction that has to be re-timed around the truth, not treated as a constraint on the train that is actually there. So in the search, the start state of an on-map train gets a free pass to hold its own cell to the horizon, and when it replans we wake every train whose reservations conflicted with that cell so that they replan around reality:

```python
# a replanning on-map train may keep its current cell indefinitely;
# wake every stale claimant so the queue learns the truth at once.
if agent.state not in OFF_MAP_STATES:
    for other in self.res.conflicting_owners(h, g, visits[:1]):
        self.dirty.add(other)
```

One detail that mattered here is that we wake all the conflicting claimants, not just the earliest one, because waking only the earliest lets the queue learn the truth one train at a time, which is too slow, and wedges still form while the news travels. If you wake all of them at once the whole queue re-times in a single step. On the worst storm environment this roughly doubled delivery on its own, from 34% up to **74.6%**.

### Fix 2: directional corridor locks

The second failure was the head-on deadlock from the swap section, but at scale. Watch two convoys come at the same long single-track artery from opposite ends. Each train on its own has a valid plan, but once the first train from each side commits into the artery they are on a collision course, and by the time the reservation conflict is visible they are already too deep to back out, because Flatland has no reverse. So they meet in the middle, cannot pass, cannot swap, and the artery is dead along with everything queued behind both of them.

The reservation table on its own does not stop this, because each train's cells are free when it reserves them, so the conflict is a property of the whole corridor and not of any single cell. So we made the planner treat corridors as first-class objects. First we detect them, where a corridor is a maximal chain of plain cells, degree at most 2 with no routing choice, and on this map there are 471 of them, and the 18 long arteries (20 cells or more) are exactly where the face-offs happen. Here is what the solver sees:

![The single-track corridors and the long arteries that get locked](assets/flatland_corridors.png)

The purple segments are the long arteries that get locked, and the teal ones are shorter corridors that resolve fine with normal cell reservations. Then the mechanism is that **any plan that traverses a long artery locks it for that plan's direction and time window**, and an opposing plan that wants the same artery in the same window has to schedule after the lock releases, while same-direction platooning stays completely free, because trains chasing each other down the artery is fine, it is only the head-on case that has to be serialized:

![A directional corridor lock in space-time](assets/flatland_corridor_lock.png)

In the search, before a train enters an artery it checks for an opposing lock over the window it wants to traverse, and if it finds one it waits at the mouth until it clears:

```python
if est_transit:                       # entering a long artery
    sense_v = g.corridor_sense[v_node]
    for _ in range(8):
        lock_end = self.res.opposing_lock_until(seg_v, sense_v, tau, tau + est_transit, agent)
        if lock_end < 0:
            break                     # no opposing lock, go
        tau = lock_end                # wait for the artery to clear its direction
```

Tuning the length threshold was a small experiment on its own. Locking every corridor over-serializes the station throats, where short parallel tracks are meant to be used at the same time, and that cost us points, while locking only the long arteries hits just the handful of places where face-offs actually form. We swept the threshold, and 12 cells was too aggressive and 20 was the sweet spot, because it covers only the roughly 18 longest arteries. Here is a real face-off, an end-of-episode close-up of one contested artery on a 62-train scenario, with the lock off and then on:

![An artery face-off at end of episode, locks off vs on](assets/flatland_faceoff_artery.png)

Those red wedges are trains stopped nose to tail in a single-track artery, each one facing traffic it can neither pass nor reverse around. Without the lock both ends of the artery fill up and 16 trains are stranded, and with the lock the opposing traffic serializes at the mouths and the wedge shrinks. This particular artery still keeps a stubborn residual of 12 stuck trains even with the lock on, and every quick fix we threw at it (escape moves, locking the artery stubs) tested worse, so we left it, because it is worth about `0.1` normalized on a synthetic environment that may not even look like the real levels. But on the 118-train storm, where these face-offs are the main failure mode, corridor locks took delivery from **74.6% to 93.2%**, which is 88 to 110 of the 118 trains home. Stacked on top of overstay rights, the two fixes together turned a 34%-delivery disaster into a 93%-delivery success.

Here is the shipped planner actually dispatching a 62-train scenario, colored by state, moving in teal and stopped in red, with most of the fleet flowing at once:

![The planner dispatching a 62-train scenario, coloured by state](assets/flatland_roll_lock20_t250.png)

Across the full local benchmark suite the final configuration looks like this:

| suite | score | delivered |
|---|---|---|
| example curriculum (max 27) | 25.35 | 100.0% |
| official-generator level-1/2 replicas | 0.879 / 0.891 | 100% |
| 532 agents, mixed speeds, no malfunction | 0.971 | 100% |
| 118-agent malfunction storm | 0.803 | 93% |
| RL baseline, for reference | 15.55 | 45.6% |

The 532-agent number is worth sitting with for a second, because the largest competition scenarios put 532 trains on this map, and the planner handles them at 97 to 100% delivery in about four minutes of wall-clock, well inside the 30-minute budget, so prioritized SIPP scales.

## Things we tried that did not work

At least half of the work was ideas that sounded good and tested worse, and we measured every one of them with one-flag-at-a-time ablations instead of trusting our gut, and that paid off again and again, because a lot of the "obviously helpful" mechanisms were actively hurting. Roughly in the order we killed them:

- **Headway buffers.** The instinct from MAPF is to keep a one-step gap between trains, but in this reward regime a headway of 1 means every train that catches up to another has to stop to keep the gap, and every stop is `-250 x v`, so headway just does not fit the braking economics. We run `headway = 0`, so trains chain nose to tail, which is free, and only stop when they are actually blocked.

- **Cascade replanning.** When a train replans and its new path conflicts with others, the tempting move is to immediately replan all of them too, but it just churns, because the same conflicts re-form, planning time blows up, and the score does not move. We only wake the specific claimants of a train's own current cell, which is the overstay case, and nothing more.

- **Relaxed fallback planning.** When the table was too crowded to find a conflict-free plan, we tried letting the train plan against only the static obstacles (stopped and broken trains) and sorting the rest out during execution, and this was a disaster, because trains committed to paths straight through moving traffic and produced exactly the head-on deadlocks we were trying to avoid. Off.

- **Shift-repair and eager-repair.** When a train is only delayed, why re-search from scratch, just push its remaining schedule later by the delay. We measured it, and it was `-0.55` and `-0.57` on the curriculum, because shifting keeps a route that a full replan would have routed around the new obstacle for free, so the full SIPP re-search is just better and both repairs are off.

- **Escape moves.** For a hopelessly stuck train, step it out of the way to break the jam. Net negative, because the escaping train just brakes (another `-250`) and parks somewhere that blocks a different route. Off.

- **Serving intermediate stops.** We tried turning stop-serving back on to collect the rebate, but it destabilizes dense scenarios, because halting at a platform is exactly what triggers the departure-delay malfunctions at levels 4 and 5, and the chaos that follows costs far more than the `-50` rebates give back. The whole point was to skip, so leaving it off was right.

- **Stub corridor locks.** We tried having plan-less stubbed trains also lock their corridor, and it measured worse both ways, so locks stay traversal-only.

The lesson here is the same one from every optimization project, which is that the reward function knows more than you do and your intuition was built on a different reward function, so you measure everything instead of trusting it.

## Repository layout

- **`submission/planner/`** — the dispatcher. [`netgraph.py`](submission/planner/netgraph.py) builds the `(cell, direction)` graph and detects the single-track corridors; [`sipp.py`](submission/planner/sipp.py) is Safe-Interval Path Planning over the shared reservation table, including overstay rights and the directional corridor locks; [`runtime.py`](submission/planner/runtime.py) is the plan-and-execute loop with priority planning, malfunction replanning, and the safety layers.
- **`submission/my_policy.py`**, **`submission/my_observation_builder.py`** — the policy and env-passing observation builder the evaluator calls.
- **`dev/`** — the local evaluation harness ([`eval_lib.py`](dev/eval_lib.py), [`evaluate.py`](dev/evaluate.py)) that mirrors the server-side scoring, plus the stress environments and per-version result CSVs.

## Reproducing locally

```bash
pip install flatland-rl==4.2.6

# score the SIPP planner on the example curriculum, with the official
# normalization and a per-penalty breakdown (mirrors the server-side eval)
python -m dev.evaluate --glob 'reinforcement_learning/curriculum/curriculum/example_curriculum/*.pkl'
```

Behaviour is controlled by environment-variable knobs, all defaulting to the measured-best configuration, for example `PLANNER_LOCK_MIN_LEN=20` for the corridor-lock length gate and `PLANNER_SERVE_STOPS=0` to keep stop-skipping on. See [`dev/SUBMISSION_NOTES.md`](dev/SUBMISSION_NOTES.md) for the full list and the tested-and-rejected ablations.

---

**References:**
1. [ECML 2026 Flatland Real-World Baselines Challenge](https://competition.flatland.cloud)
2. [Flatland environment and evaluation code (flatland-rl)](https://github.com/flatland-association/flatland-rl)
3. [Phillips and Likhachev, *SIPP: Safe Interval Path Planning for Dynamic Environments*, ICRA 2011](https://www.cs.cmu.edu/~maxim/files/sipp_icra11.pdf)
4. [Silver, *Cooperative Pathfinding* (prioritized planning with reservation tables), AIIDE 2005](https://www.davidsilver.uk/wp-content/uploads/2020/03/coop-path-AIIDEo5.pdf)
