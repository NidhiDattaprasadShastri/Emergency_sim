import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

LOAD_MIN = 3                                                  
TREATMENT_MIN = {"immediate": 90, "delayed": 60, "minor": 30}  
RUN_UNTIL_MIN = 240                                           
TEAM_WORK_MIN = 10                                            
STAFF_TRANSFER_MIN = 15                                       
SURGE_BEDS = {"H-A": 2, "H-B": 2, "H-C": 1}                   
BEDS_PER_LOAN = 2                                           

@dataclass
class AssignResponder:
    team_id: str
    incident_id: str
    reason: str = ""


@dataclass
class LendStaff:
    from_hospital: str
    to_hospital: str
    reason: str = ""


@dataclass
class Dispatch:
    ambulance_id: str
    incident_id: str
    victim_ids: list
    hospital_id: str
    reason: str = ""

def manhattan_km(a, b):
    return abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])


@dataclass
class Simulation:
    world: dict
    t: int = 0
    log: list = field(default_factory=list)      
    snapshots: list = field(default_factory=list) 

    def __post_init__(self):
        w = self.world
        self.rng = np.random.default_rng(w["seed"] + 1000) 
        self.speed = w["ambulance_speed_kmh"]
        self.traffic = w["traffic_factor"]

        self.incidents = {i["incident_id"]: dict(i, responder_on_scene=False, team_id=None,
                                                 first_unit_min=None) for i in w["incidents"]}
        self.victims = {}
        for inc in w["incidents"]:
            for v in inc["victims"]:
                self.victims[v["victim_id"]] = dict(v, incident_id=inc["incident_id"],
                                                    status="not_reported", 
                                                    admitted_min=None, hospital_id=None)
        self.ambulances = {a["ambulance_id"]: dict(a, status="available", plan=None, busy_min=0)
                           for a in w["ambulances"]}
        self.teams = {r["team_id"]: dict(r, status="available", plan=None) for r in w["responders"]}
        self.hospitals = {h["hospital_id"]: dict(h, beds_available=h["er_beds"], beds_total=h["er_beds"],
                                                 surge_open=0) for h in w["hospitals"]}
        self.staff = {s["staff_id"]: dict(s, current_load=0) for s in w["staff"]}
        self.treatments = []         
        self.transfers = []          
        self.staff_loans = 0
        self.rejected = 0

    def travel_min(self, a, b):
        factor = self.rng.uniform(*self.traffic)
        return max(1, int(round(manhattan_km(a, b) / self.speed * 60 * factor)))

    def estimate_min(self, a, b):
        return manhattan_km(a, b) / self.speed * 60

    def waiting_victims(self, incident_id=None):
        return [v for v in self.victims.values() if v["status"] == "waiting"
                and (incident_id is None or v["incident_id"] == incident_id)]

    def active_incidents(self):
        return [i for i in self.incidents.values() if self.waiting_victims(i["incident_id"])]

    def free_staff(self, hospital_id, role, skill=None):
        return [s for s in self.staff.values() if s["hospital_id"] == hospital_id and s["role"] == role
                and s["current_load"] < s["workload_cap"] and (skill is None or skill in s["skills"])]

    def idle_staff(self, hospital_id, role):
        return [s for s in self.staff.values() if s["hospital_id"] == hospital_id and s["role"] == role
                and s["current_load"] == 0]

    def can_lend(self, from_h, to_h):
        """A hospital can lend one idle doctor + nurse if it keeps enough staff slots for all its own
        beds afterwards, and the other hospital still has surge beds to open."""
        h = self.hospitals[from_h]
        if h.get("diverted") or not self.idle_staff(from_h, "doctor") or not self.idle_staff(from_h, "nurse"):
            return False
        doc_cap = sum(s["workload_cap"] for s in self.staff.values()
                      if s["hospital_id"] == from_h and s["role"] == "doctor")
        nurse_cap = sum(s["workload_cap"] for s in self.staff.values()
                        if s["hospital_id"] == from_h and s["role"] == "nurse")
        d = self.idle_staff(from_h, "doctor")[0]["workload_cap"]
        n = self.idle_staff(from_h, "nurse")[0]["workload_cap"]
        return (doc_cap - d >= h["beds_total"] and nurse_cap - n >= h["beds_total"]
                and self.surge_left(to_h) > 0 and not self.hospitals[to_h].get("diverted"))

    def surge_left(self, hospital_id):
        h = self.hospitals[hospital_id]
        pending = sum(BEDS_PER_LOAN for x in self.transfers if x[2] == hospital_id)
        return max(0, SURGE_BEDS.get(hospital_id, 0) - h["surge_open"] - pending)

    def hospital_capacity(self, hospital_id):
        """How many more patients a hospital can take right now (beds AND staff)."""
        h = self.hospitals[hospital_id]
        if h.get("diverted"):
            return 0                      
        doc_slots = sum(s["workload_cap"] - s["current_load"] for s in self.free_staff(hospital_id, "doctor"))
        nurse_slots = sum(s["workload_cap"] - s["current_load"] for s in self.free_staff(hospital_id, "nurse"))
        return max(0, min(h["beds_available"], doc_slots, nurse_slots))

    def apply(self, actions, policy_name=""):
        for a in actions:
            if isinstance(a, AssignResponder):
                ok, why = self._assign_responder(a)
            elif isinstance(a, LendStaff):
                ok, why = self._lend_staff(a)
            else:
                ok, why = self._dispatch(a)
            self.log.append({"t": self.t, "type": "action_accepted" if ok else "action_rejected",
                             "action": type(a).__name__, **a.__dict__, "note": why, "policy": policy_name})
            if not ok:
                self.rejected += 1

    def _assign_responder(self, a):
        team, inc = self.teams.get(a.team_id), self.incidents.get(a.incident_id)
        if not team or not inc:
            return False, "unknown team or incident"
        if team["status"] != "available":
            return False, "team busy"
        if inc["team_id"]:
            return False, "incident already has a team"
        arrive = self.t + self.travel_min(team["location"], inc["location"])
        team.update(status="en_route", plan={"incident_id": a.incident_id, "arrive": arrive})
        inc["team_id"] = a.team_id
        return True, f"arrives at minute {arrive}"

    def _lend_staff(self, a):
        if a.from_hospital not in self.hospitals or a.to_hospital not in self.hospitals \
                or a.from_hospital == a.to_hospital:
            return False, "unknown hospital"
        if not self.can_lend(a.from_hospital, a.to_hospital):
            return False, "no idle staff to spare, or no surge beds left"
        pair = [self.idle_staff(a.from_hospital, "doctor")[0], self.idle_staff(a.from_hospital, "nurse")[0]]
        for s in pair:
            s["hospital_id"] = "in_transit"
        self.transfers.append((self.t + STAFF_TRANSFER_MIN, [s["staff_id"] for s in pair], a.to_hospital))
        self.staff_loans += 1
        return True, f"{pair[0]['staff_id']} + {pair[1]['staff_id']} arrive at {a.to_hospital} minute {self.t + STAFF_TRANSFER_MIN}"

    def _dispatch(self, a):
        amb, inc = self.ambulances.get(a.ambulance_id), self.incidents.get(a.incident_id)
        if not amb or not inc or a.hospital_id not in self.hospitals:
            return False, "unknown ambulance, incident or hospital"
        if amb["status"] != "available":
            return False, "ambulance busy"
        if not a.victim_ids or len(a.victim_ids) > amb["capacity"]:
            return False, "too many or zero victims for this ambulance"
        for vid in a.victim_ids:
            v = self.victims.get(vid)
            if not v or v["status"] != "waiting" or v["incident_id"] != a.incident_id:
                return False, f"victim {vid} not waiting at this incident"
        if self.hospital_capacity(a.hospital_id) < len(a.victim_ids):
            return False, "hospital has no capacity"

        h = self.hospitals[a.hospital_id]
        h["beds_available"] -= len(a.victim_ids)
        reserved = []
        for vid in a.victim_ids:
            sev = self.victims[vid]["severity"]
            doc = (self.free_staff(a.hospital_id, "doctor", "trauma") if sev == "immediate" else []) \
                  or self.free_staff(a.hospital_id, "doctor")
            nurse = self.free_staff(a.hospital_id, "nurse")
            for s in (doc[0], nurse[0]):
                s["current_load"] += 1
            reserved.append([doc[0]["staff_id"], nurse[0]["staff_id"]])
            self.victims[vid]["status"] = "assigned"

        to_scene = self.travel_min(amb["location"], inc["location"])
        to_hosp = self.travel_min(inc["location"], h["location"])
        amb.update(status="en_route", plan={
            "incident_id": a.incident_id, "victim_ids": a.victim_ids, "hospital_id": a.hospital_id,
            "staff": reserved, "at_scene": self.t + to_scene, "to_hosp": to_hosp,
            "depart": None, "arrive_hosp": None})
        return True, f"at scene minute {self.t + to_scene}"

    def advance(self):
        self.t += 1
        t = self.t

        for ev in self.world.get("events", []):
            if ev["t"] == t and ev["type"] == "diversion":
                self.hospitals[ev["hospital_id"]]["diverted"] = True
                self.log.append({"t": t, "type": "hospital_diversion", "hospital_id": ev["hospital_id"]})

        for item in [x for x in self.transfers if x[0] <= t]:
            _, staff_ids, hid = item
            h = self.hospitals[hid]
            for sid in staff_ids:
                self.staff[sid]["hospital_id"] = hid
            opened = min(BEDS_PER_LOAN, SURGE_BEDS.get(hid, 0) - h["surge_open"])
            h["surge_open"] += opened
            h["beds_total"] += opened
            h["beds_available"] += opened
            self.transfers.remove(item)
            self.log.append({"t": t, "type": "staff_arrived", "hospital_id": hid, "staff_ids": staff_ids,
                             "surge_beds_opened": opened})

        for inc in self.incidents.values():
            if inc["arrival_min"] == t:
                for v in inc["victims"]:
                    self.victims[v["victim_id"]]["status"] = "waiting"
                self.log.append({"t": t, "type": "incident_reported", "incident_id": inc["incident_id"],
                                 "victims": len(inc["victims"]), "report": inc.get("report", "")})
        for team in self.teams.values():
            p = team["plan"]
            if team["status"] == "en_route" and t >= p["arrive"]:
                inc = self.incidents[p["incident_id"]]
                team.update(status="on_scene", location=dict(inc["location"]))
                p["done"] = t + TEAM_WORK_MIN
                inc["first_unit_min"] = inc["first_unit_min"] or t
                self.log.append({"t": t, "type": "team_on_scene", "team_id": team["team_id"],
                                 "incident_id": inc["incident_id"]})
            elif team["status"] == "on_scene" and t >= p["done"]:
                self.incidents[p["incident_id"]]["responder_on_scene"] = True  
                team.update(status="available", plan=None)
                self.log.append({"t": t, "type": "scene_cleared", "team_id": team["team_id"],
                                 "incident_id": p["incident_id"]})

        for amb in self.ambulances.values():
            p = amb["plan"]
            if amb["status"] != "available":
                amb["busy_min"] += 1
            if amb["status"] == "en_route" and t >= p["at_scene"]:
                inc = self.incidents[p["incident_id"]]
                amb["location"] = dict(inc["location"])
                inc["first_unit_min"] = inc["first_unit_min"] or t
                needs_team = bool(inc["needs_responders"])
                if not needs_team or inc["responder_on_scene"]:
                    amb["status"] = "loading"
                    p["depart"] = t + LOAD_MIN
                else:
                    amb["status"] = "waiting_for_responders"
            elif amb["status"] == "waiting_for_responders" and self.incidents[p["incident_id"]]["responder_on_scene"]:
                amb["status"] = "loading"
                p["depart"] = t + LOAD_MIN
            elif amb["status"] == "loading" and t >= p["depart"]:
                amb["status"] = "transporting"
                p["arrive_hosp"] = t + p["to_hosp"]
                for vid in p["victim_ids"]:
                    self.victims[vid]["status"] = "in_transport"
            elif amb["status"] == "transporting" and t >= p["arrive_hosp"]:
                h = self.hospitals[p["hospital_id"]]
                for vid, staff_ids in zip(p["victim_ids"], p["staff"]):
                    v = self.victims[vid]
                    v.update(status="admitted", admitted_min=t, hospital_id=h["hospital_id"])
                    self.treatments.append((t + TREATMENT_MIN[v["severity"]], vid, h["hospital_id"], staff_ids))
                self.log.append({"t": t, "type": "patients_admitted", "ambulance_id": amb["ambulance_id"],
                                 "hospital_id": h["hospital_id"], "victim_ids": p["victim_ids"]})
                amb.update(status="available", plan=None, location=dict(h["location"]))

        for item in [x for x in self.treatments if x[0] <= t]:
            _, vid, hid, staff_ids = item
            self.hospitals[hid]["beds_available"] += 1
            for sid in staff_ids:
                self.staff[sid]["current_load"] -= 1
            self.victims[vid]["status"] = "discharged"
            self.treatments.remove(item)

        for v in self.victims.values():
            if (v["severity"] == "immediate" and v["deteriorates_at_min"] == t
                    and v["status"] in ("waiting", "assigned", "in_transport")):
                self.log.append({"t": t, "type": "victim_deteriorated", "victim_id": v["victim_id"],
                                 "incident_id": v["incident_id"]})

        self.snapshots.append(self._snapshot())

    def _snapshot(self):
        return {"t": self.t,
                "ambulances": {k: a["status"] for k, a in self.ambulances.items()},
                "teams": {k: x["status"] for k, x in self.teams.items()},
                "beds_available": {k: h["beds_available"] for k, h in self.hospitals.items()},
                "beds_total": {k: h["beds_total"] for k, h in self.hospitals.items()},
                "staff_load": {k: sum(s["current_load"] for s in self.staff.values() if s["hospital_id"] == k)
                               for k in self.hospitals},
                "doctors": {k: self._role_load(k, "doctor") for k in self.hospitals},
                "nurses": {k: self._role_load(k, "nurse") for k in self.hospitals},
                "waiting": len(self.waiting_victims())}

    def _role_load(self, hospital_id, role):
        """[busy slots, total slots, people] for one role at one hospital."""
        staff = [s for s in self.staff.values() if s["hospital_id"] == hospital_id and s["role"] == role]
        return [sum(s["current_load"] for s in staff), sum(s["workload_cap"] for s in staff), len(staff)]

    def run(self, policy, policy_name="policy"):
        while self.t < RUN_UNTIL_MIN:
            self.advance()
            actions = policy(self)
            if actions:
                self.apply(actions, policy_name)
        return self.metrics()

    def metrics(self):
        vs = list(self.victims.values())
        served = [v for v in vs if v["admitted_min"] is not None]
        critical = [v for v in vs if v["severity"] == "immediate"]
        crit_late = [v for v in critical if v["admitted_min"] is None or v["admitted_min"] > v["deteriorates_at_min"]]
        resp = [i["first_unit_min"] - i["arrival_min"] for i in self.incidents.values() if i["first_unit_min"]]
        crit_wait = [v["admitted_min"] - self.incidents[v["incident_id"]]["arrival_min"] for v in critical
                     if v["admitted_min"] is not None]
        amb_util = sum(a["busy_min"] for a in self.ambulances.values()) / (len(self.ambulances) * self.t)
        return {
            "victims_total": len(vs),
            "victims_served": len(served),
            "critical_total": len(critical),
            "critical_unserved_in_time": len(crit_late),
            "avg_response_min": round(float(np.mean(resp)), 1) if resp else None,
            "avg_critical_to_hospital_min": round(float(np.mean(crit_wait)), 1) if crit_wait else None,
            "ambulance_utilization": round(amb_util, 3),
            "rejected_actions": self.rejected,
            "staff_loans": self.staff_loans,
        }

    def save(self, path, policy_name, extra=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"policy": policy_name, "seed": self.world["seed"],
                                          "metrics": self.metrics(), "log": self.log,
                                          "snapshots": self.snapshots, **(extra or {})}, indent=1))

def load_world(path):
    return json.loads(Path(path).read_text())