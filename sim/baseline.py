from sim.engine import AssignResponder, Dispatch


def greedy_policy(sim):
    actions = []
    used_ambulances, used_teams = set(), set()
    reserved = {h: sim.hospital_capacity(h) for h in sim.hospitals}

    for inc in sorted(sim.active_incidents(), key=lambda i: i["arrival_min"]):
        for need in inc["needs_responders"]:
            if inc["team_id"] is None:
                teams = [t for t in sim.teams.values() if t["team_type"] == need
                         and t["status"] == "available" and t["team_id"] not in used_teams]
                if teams:
                    team = min(teams, key=lambda t: sim.estimate_min(t["location"], inc["location"]))
                    actions.append(AssignResponder(team["team_id"], inc["incident_id"], "nearest team"))
                    used_teams.add(team["team_id"])

        waiting = sim.waiting_victims(inc["incident_id"])
        while waiting:
            free = [a for a in sim.ambulances.values()
                    if a["status"] == "available" and a["ambulance_id"] not in used_ambulances]
            if not free:
                return actions
            amb = min(free, key=lambda a: sim.estimate_min(a["location"], inc["location"]))
            load = waiting[: amb["capacity"]]
            hosps = [h for h in sim.hospitals.values() if reserved[h["hospital_id"]] >= len(load)]
            if not hosps:
                return actions
            hosp = min(hosps, key=lambda h: sim.estimate_min(inc["location"], h["location"]))
            actions.append(Dispatch(amb["ambulance_id"], inc["incident_id"],
                                    [v["victim_id"] for v in load], hosp["hospital_id"],
                                    "nearest ambulance, nearest hospital"))
            used_ambulances.add(amb["ambulance_id"])
            reserved[hosp["hospital_id"]] -= len(load)
            waiting = waiting[len(load):]
    return actions
