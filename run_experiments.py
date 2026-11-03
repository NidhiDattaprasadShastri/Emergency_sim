import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from generate_world import make_incidents, make_resources
from narrate import _facts, _template_report
from sim.baseline import greedy_policy
from sim.engine import Simulation
from team.llm import LLM
from team.team import AgentTeam

SCENARIOS = ["normal", "surge", "diversion"]
KEY_METRICS = ["victims_served", "critical_unserved_in_time", "avg_response_min",
               "avg_critical_to_hospital_min", "ambulance_utilization"]


def build_world(cfg, seed):
    rng = np.random.default_rng(seed)
    hospitals, staff, ambulances, responders = make_resources(rng, cfg)
    incidents = make_incidents(rng, cfg)
    for inc in incidents:
        inc["report"] = _template_report(_facts(inc))
    return {"seed": seed, "grid_km": cfg["grid_km"], "horizon_min": cfg["horizon_min"],
            "ambulance_speed_kmh": cfg["ambulance_speed_kmh"], "traffic_factor": cfg["traffic_factor"],
            "hospitals": hospitals, "staff": staff, "ambulances": ambulances,
            "responders": responders, "incidents": incidents, "events": []}


def apply_scenario(world, scenario, cfg):
    w = copy.deepcopy(world)
    if scenario == "surge":
        for k, inc in enumerate(w["incidents"][2:8]):
            inc["arrival_min"] = 40 + k * 3
        w["incidents"].sort(key=lambda i: i["arrival_min"])
        for inc in w["incidents"]:     
            for v in inc["victims"]:
                limit = cfg["deterioration_min"].get(v["severity"])
                v["deteriorates_at_min"] = inc["arrival_min"] + limit if limit else None
    elif scenario == "diversion":
        w["events"].append({"t": 60, "type": "diversion", "hospital_id": "H-A"})
    return w


def run_one(world, policy_name):
    sim = Simulation(world)
    if policy_name == "baseline":
        metrics = sim.run(greedy_policy, "baseline")
        extra = {}
    else:
        team = AgentTeam(LLM(enabled=False), mode="rule")
        metrics = sim.run(team, "agents")
        extra = {"contested_ambulances": team.stats["contested_ambulances"],
                 "counters": team.stats["counters"], "rejects": team.stats["rejects"],
                 "staff_requests": team.stats["staff_requests"]}
    return metrics | extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    cfg = json.loads(Path("config.json").read_text())

    rows = []
    for scenario in SCENARIOS:
        for seed in range(1, args.seeds + 1):
            world = apply_scenario(build_world(cfg, seed), scenario, cfg)
            for policy in ("baseline", "agents"):
                m = run_one(world, policy)
                rows.append({"scenario": scenario, "seed": seed, "policy": policy, **m})
            print(f"  {scenario:9} seed {seed:2}  done", flush=True)

    Path("results").mkdir(exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open("results/experiments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    print("\nAverage over seeds (baseline -> agents)")
    print(f"  {'scenario':10} {'metric':30} {'baseline':>10} {'agents':>10} {'change':>9}  agents better in")
    for scenario in SCENARIOS:
        for metric in KEY_METRICS:
            b = [r[metric] for r in rows if r["scenario"] == scenario and r["policy"] == "baseline"]
            a = [r[metric] for r in rows if r["scenario"] == scenario and r["policy"] == "agents"]
            pairs = [(x, y) for x, y in zip(b, a) if x is not None and y is not None]
            bm, am = np.mean([p[0] for p in pairs]), np.mean([p[1] for p in pairs])
            higher_is_better = metric in ("victims_served", "ambulance_utilization")
            wins = sum((y > x) if higher_is_better else (y < x) for x, y in pairs)
            summary.append({"scenario": scenario, "metric": metric, "baseline_mean": round(bm, 2),
                            "baseline_std": round(float(np.std([p[0] for p in pairs])), 2),
                            "agents_mean": round(am, 2),
                            "agents_std": round(float(np.std([p[1] for p in pairs])), 2),
                            "agents_better_in": f"{wins}/{len(pairs)}"})
            d = 2 if metric == "ambulance_utilization" else 1
            print(f"  {scenario:10} {metric:30} {bm:10.{d}f} {am:10.{d}f} {am - bm:+9.{d}f}  {wins}/{len(pairs)} seeds")
        print()

    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("Saved results/experiments.csv and results/summary.csv")


if __name__ == "__main__":
    main()