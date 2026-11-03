import json
import re
from pathlib import Path

SEV_RANK = {"immediate": 0, "delayed": 1, "minor": 2}
AGENT_OF = {"Assessment": "Incident Assessment", "Dispatch": "Dispatch",
            "Allocation": "Resource Allocation"}   


def list_runs(folder="runs"):
    return sorted(str(p) for p in Path(folder).glob("*.json"))


def load_json(path):
    return json.loads(Path(path).read_text())


def agent_name(sender):
    return AGENT_OF.get(sender, "Hospital Coordination" if sender.startswith("H-") else sender)


def lerp(a, b, f):
    f = max(0.0, min(1.0, f))
    return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f


def xy(loc):
    return (loc["x"], loc["y"])


def build_timeline(run, world):
    snaps = run["snapshots"]
    T = len(snaps)
    log = run["log"]
    inc_loc = {i["incident_id"]: xy(i["location"]) for i in world["incidents"]}
    hosp_loc = {h["hospital_id"]: xy(h["location"]) for h in world["hospitals"]}

    jobs = {a["ambulance_id"]: [] for a in world["ambulances"]}
    admitted_at = {}
    for e in log:
        if e["type"] == "action_accepted" and e["action"] == "Dispatch":
            m = re.search(r"minute (\d+)", e.get("note", ""))
            jobs[e["ambulance_id"]].append({"t0": e["t"], "scene": int(m.group(1)) if m else e["t"] + 10,
                                            "incident_id": e["incident_id"], "hospital_id": e["hospital_id"],
                                            "victim_ids": e["victim_ids"], "admit": None, "transport": None})
        if e["type"] == "patients_admitted":
            for j in jobs[e["ambulance_id"]]:
                if j["admit"] is None and j["victim_ids"] == e["victim_ids"]:
                    j["admit"] = e["t"]
                    break
            for v in e["victim_ids"]:
                admitted_at[v] = e["t"]
    for aid, js in jobs.items():
        for j in js:
            j["transport"] = next((t for t in range(j["t0"] + 1, (j["admit"] or T) + 1)
                                   if t <= T and snaps[t - 1]["ambulances"][aid] == "transporting"), None)

    amb_pos = {}   
    for aid in jobs:
        start = xy(next(a for a in world["ambulances"] if a["ambulance_id"] == aid)["location"])
        k, here = 0, start
        for t in range(1, T + 1):
            js = jobs[aid]
            while k < len(js) and js[k]["admit"] is not None and t > js[k]["admit"]:
                here = hosp_loc[js[k]["hospital_id"]]
                k += 1
            status = snaps[t - 1]["ambulances"][aid]
            j = js[k] if k < len(js) and t > js[k]["t0"] else None
            target, pos = "", here
            if j and status == "en_route":
                pos = lerp(here, inc_loc[j["incident_id"]], (t - j["t0"]) / max(1, j["scene"] - j["t0"]))
                target = j["incident_id"]
            elif j and status in ("waiting_for_responders", "loading"):
                pos, target = inc_loc[j["incident_id"]], j["incident_id"]
            elif j and status == "transporting":
                ts, ta = j["transport"] or t, j["admit"] or T
                pos = lerp(inc_loc[j["incident_id"]], hosp_loc[j["hospital_id"]], (t - ts) / max(1, ta - ts))
                target = j["hospital_id"]
            amb_pos.setdefault(t, {})[aid] = (pos[0], pos[1], status, target)

    team_pos = {}
    tjobs = {r["team_id"]: [] for r in world["responders"]}
    for e in log:
        if e["type"] == "action_accepted" and e["action"] == "AssignResponder":
            m = re.search(r"minute (\d+)", e.get("note", ""))
            tjobs[e["team_id"]].append({"t0": e["t"], "arrive": int(m.group(1)) if m else e["t"] + 10,
                                        "incident_id": e["incident_id"]})
    for tid, js in tjobs.items():
        here = xy(next(r for r in world["responders"] if r["team_id"] == tid)["location"])
        for t in range(1, T + 1):
            status = snaps[t - 1]["teams"][tid]
            past = [j for j in js if j["t0"] < t]
            j = past[-1] if past else None
            pos, target = here, ""
            if j:
                start = here if len(past) == 1 else inc_loc[past[-2]["incident_id"]]
                if status == "en_route":
                    pos = lerp(start, inc_loc[j["incident_id"]], (t - j["t0"]) / max(1, j["arrive"] - j["t0"]))
                    target = j["incident_id"]
                else:
                    pos = inc_loc[j["incident_id"]]
                    target = j["incident_id"] if status == "on_scene" else ""
            team_pos.setdefault(t, {})[tid] = (pos[0], pos[1], status, target)

    picked_at = {}
    for js in jobs.values():
        for j in js:
            for v in j["victim_ids"]:
                picked_at[v] = j["t0"]
    victims = [dict(v, incident_id=i["incident_id"], arrival=i["arrival_min"])
               for i in world["incidents"] for v in i["victims"]]

    def incidents_at(t):
        rows = []
        for i in world["incidents"]:
            if i["arrival_min"] > t:
                continue
            waiting = [v for v in i["victims"] if picked_at.get(v["victim_id"], 10 ** 9) >= t]
            worst = min((SEV_RANK[v["severity"]] for v in waiting), default=None)
            rows.append({"incident_id": i["incident_id"], "x": i["location"]["x"], "y": i["location"]["y"],
                         "type": i["incident_type"], "victims": len(i["victims"]), "waiting": len(waiting),
                         "worst": ["immediate", "delayed", "minor"][worst] if worst is not None else "cleared",
                         "report": i.get("report", "")})
        return rows

    def kpis_at(t):
        reported = [v for v in victims if v["arrival"] <= t]
        served = sum(1 for v in reported if admitted_at.get(v["victim_id"], 10 ** 9) <= t)
        missed = sum(1 for v in reported if v["severity"] == "immediate" and v["deteriorates_at_min"] <= t
                     and admitted_at.get(v["victim_id"], 10 ** 9) > v["deteriorates_at_min"])
        return {"reported": len(reported), "served": served, "waiting": snaps[t - 1]["waiting"],
                "critical_missed": missed}

    return {"T": T, "amb": amb_pos, "teams": team_pos, "incidents_at": incidents_at,
            "kpis_at": kpis_at, "hospitals": world["hospitals"]}