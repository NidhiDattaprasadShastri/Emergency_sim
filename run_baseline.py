import argparse

from sim.baseline import greedy_policy
from sim.engine import Simulation, load_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    world = load_world(f"data/world_v1.0_seed{args.seed}.json")
    sim = Simulation(world)
    metrics = sim.run(greedy_policy, "baseline")
    sim.save(f"runs/baseline_seed{args.seed}.json", "baseline")

    print(f"Baseline results, seed {args.seed}")
    for k, v in metrics.items():
        print(f"  {k:30} {v}")
    print(f"\nSaved runs/baseline_seed{args.seed}.json ({len(sim.log)} events)")


if __name__ == "__main__":
    main()
