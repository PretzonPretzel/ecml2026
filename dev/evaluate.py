"""
Evaluate a policy over a set of scenario pkls, in parallel.

Usage:
  .venv/bin/python -m dev.evaluate --glob 'reinforcement_learning/curriculum/curriculum/example_curriculum/*.pkl' \
      --policy submission.my_policy.MyPolicy --obs-builder submission.my_observation_builder.MyObservationBuilder \
      --jobs 9 --out dev/results/run1.csv
"""
import argparse
import csv
import glob as globlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dev.eval_lib import _worker  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="reinforcement_learning/curriculum/curriculum/example_curriculum/*.pkl")
    p.add_argument("--policy", default="submission.my_policy.MyPolicy")
    p.add_argument("--obs-builder", default="submission.my_observation_builder.MyObservationBuilder")
    p.add_argument("--jobs", type=int, default=9)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-malfunction", action="store_true")
    args = p.parse_args()

    envs = sorted(globlib.glob(args.glob))
    if not envs:
        print(f"no envs match {args.glob}", file=sys.stderr)
        sys.exit(1)

    jobs = [(e, args.policy, args.obs_builder, args.max_steps, False, args.no_malfunction) for e in envs]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            if "error" in row:
                print(f"[FAIL] {row['env']}: {row['error']}")
                if row.get("trace"):
                    print(row["trace"])
            else:
                print(f"[done] {row['env']:28s} norm={row['normalized_reward']:7.4f} "
                      f"done={row['pct_done']:6.2%} halts={row['halts']:4d} "
                      f"({row['run_s']:.0f}s)")

    rows.sort(key=lambda r: r["env"])
    ok = [r for r in rows if "error" not in r]
    if ok:
        total = sum(r["normalized_reward"] for r in ok)
        mean_done = sum(r["pct_done"] for r in ok) / len(ok)
        print("-" * 80)
        print(f"SUM normalized reward over {len(ok)} envs: {total:.4f}   mean %done: {mean_done:.2%}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        keys = sorted({k for r in rows for k in r if k != "trace"})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
