import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models import World
from narrate import add_reports

VERSION = "1.0"
SEVERITIES = ["immediate", "delayed", "minor"]


def loc(x, y):
    return {"x": round(float(x), 2), "y": round(float(y), 2)}


def jitter(rng, point, km=0.5, grid=20):
    return loc(np.clip(point[0] + rng.uniform(-km, km), 0, grid),
               np.clip(point[1] + rng.uniform(-km, km), 0, grid))

def poisson_arrivals(rng, rate_per_hour, horizon_min):
    t, times = 0.0, []
    mean_gap = 60.0 / rate_per_hour
    while True:
        t += rng.exponential(mean_gap)
        if t > horizon_min:
            return times
        times.append(int(round(t)))


def make_incidents(rng, cfg):
    while True:
        times = poisson_arrivals(rng, cfg["incident_rate_per_hour"], cfg["horizon_min"])
        if cfg["min_incidents"] <= len(times) <= cfg["max_incidents"]:
            break

    types = list(cfg["incident_types"])
    weights = np.array([cfg["incident_types"][t]["weight"] for t in types])
    probs = [cfg["severity_probs"][s] for s in SEVERITIES]
    lo, hi = cfg["victims_per_incident"]

    incidents = []
    for n, t in enumerate(times, start=1):
        itype = types[rng.choice(len(types), p=weights / weights.sum())]
        n_victims = int(rng.integers(lo, hi + 1))
        victims = []
        for v in range(1, n_victims + 1):
            sev = SEVERITIES[rng.choice(3, p=probs)]
            limit = cfg["deterioration_min"].get(sev)
            victims.append({"victim_id": f"INC-{n:02d}-V{v}", "severity": sev,
                            "deteriorates_at_min": t + limit if limit else None})
        worst = min(SEVERITIES.index(v["severity"]) for v in victims) + 1
        incidents.append({
            "incident_id": f"INC-{n:02d}", "arrival_min": t,
            "location": loc(*rng.uniform(0, cfg["grid_km"], 2)),
            "incident_type": itype,
            "needs_responders": cfg["incident_types"][itype]["needs"],
            "victims": victims, "priority_hint": worst,
        })
    return incidents


def make_resources(rng, cfg):
    hospitals, staff = [], []
    for h in cfg["hospitals"]:
        hospitals.append({k: h[k] for k in ("hospital_id", "name", "kind", "er_beds", "specialties")}
                         | {"location": loc(*h["location"]), "beds_available": h["er_beds"]})
        for role, count in (("doctor", h["doctors"]), ("nurse", h["nurses"])):
            for i in range(1, count + 1):
                k = int(rng.integers(1, len(h["specialties"]) + 1))
                skills = sorted(rng.choice(h["specialties"], size=k, replace=False).tolist())
                staff.append({"staff_id": f"{h['hospital_id']}-{role[0].upper()}{i}", "role": role,
                              "hospital_id": h["hospital_id"], "skills": skills,
                              "workload_cap": cfg["workload_cap"][role]})

    stations = cfg["stations"]
    n_als = round(cfg["ambulance_count"] * cfg["ambulance_als_share"])
    ambulances = [{"ambulance_id": f"AMB-{i}", "level": "ALS" if i <= n_als else "BLS",
                   "location": jitter(rng, stations[(i - 1) % len(stations)], grid=cfg["grid_km"]),
                   "capacity": cfg["ambulance_capacity"]}
                  for i in range(1, cfg["ambulance_count"] + 1)]
    responders = [{"team_id": f"TEAM-{t.upper()}", "team_type": t,
                   "location": jitter(rng, stations[i % len(stations)], grid=cfg["grid_km"])}
                  for i, t in enumerate(cfg["responder_types"])]
    return hospitals, staff, ambulances, responders


def travel_time_min(a, b, speed_kmh, traffic_factor):
    km = abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
    return km / speed_kmh * 60 * traffic_factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else cfg["seed"]
    rng = np.random.default_rng(seed)
    print(f"[1/4] Generating world with seed {seed}")
    hospitals, staff, ambulances, responders = make_resources(rng, cfg)
    incidents = make_incidents(rng, cfg)

    print("[2/4] Writing incident reports")
    add_reports(incidents, seed, use_llm=cfg["llm"]["enabled"] and not args.no_llm,
                model=cfg["llm"]["model"])

    print("[3/4] Validating with Pydantic")
    world = World(version=VERSION, seed=seed, grid_km=cfg["grid_km"], horizon_min=cfg["horizon_min"],
                  ambulance_speed_kmh=cfg["ambulance_speed_kmh"],
                  traffic_factor=tuple(cfg["traffic_factor"]),
                  hospitals=hospitals, staff=staff, ambulances=ambulances,
                  responders=responders, incidents=incidents)

    Path("data").mkdir(exist_ok=True)
    out = Path("data") / f"world_v{VERSION}_seed{seed}.json"
    out.write_text(world.model_dump_json(indent=2))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    print(f"[4/4] Saved {out}  (fingerprint {digest})")

    victims = [v for i in world.incidents for v in i.victims]
    count = lambda s: sum(v.severity == s for v in victims)
    print(f"\n{len(world.incidents)} incidents, {len(victims)} victims "
          f"({count('immediate')} immediate, {count('delayed')} delayed, {count('minor')} minor)")
    print(f"{len(world.hospitals)} hospitals, {sum(h.er_beds for h in world.hospitals)} ER beds, "
          f"{len(world.staff)} staff, {len(world.ambulances)} ambulances, {len(world.responders)} teams")
    first = world.incidents[0]
    print(f"\nExample: {first.incident_id} at minute {first.arrival_min} ({first.incident_type})\n"
          f"  Report: {first.report}")


if __name__ == "__main__":
    main()
