import math
import re

from sim.engine import AssignResponder, Dispatch, LendStaff
from team import schemas as S
from team import tools as T
from team.messages import Message

SEV = {"immediate": 0, "delayed": 1, "minor": 2}
MAX_ROUNDS = 3

ASSESS_INSTR = """You are the Incident Assessment Agent in an emergency response simulation.
You read free-text incident reports and estimate urgency. Count victims by START triage:
immediate = life-threatening (unresponsive, not breathing, severe bleeding, trapped and critical);
delayed = serious but stable (fractures, moderate burns); minor = walking wounded.
priority: 1 if any immediate victim, 2 if any delayed, else 3.
Tool: triage_keywords(incident_id) gives a keyword-based second opinion; use it if a report is unclear.
Reply with ONLY JSON: {"assessments": [{"incident_id": "...", "priority": 1, "immediate": 0,
"delayed": 0, "minor": 0, "reason": "short"}]} with one entry per incident given."""

DISPATCH_INSTR = """You are the Dispatch Agent. Your goal: get help to the most urgent victims fast.
You get active incidents (priority 1 = most urgent, waiting victims, minutes waited, whether a
fire/rescue/hazmat team is needed) and free ambulances with travel minutes (eta) to each incident.
Propose ambulances for incidents (an ambulance carries 2 victims; ALS ambulances suit priority 1)
and free responder teams of the needed type. Do not send an ambulance to an incident that needs a
team but has none assigned (it would only wait). Beds are scarce: while serious_waiting > 0
anywhere, do not use ambulances for incidents with only minor victims. Advocate for each incident: you MAY propose the same
ambulance for two incidents; the Resource Allocation Agent will settle the clash.
Tools: travel_time(unit_id, incident_id) and incident_status(incident_id) give live facts.
Reply with ONLY JSON: {"ambulance_proposals": [{"ambulance_id": "...", "incident_id": "...",
"reason": "short"}], "team_proposals": [{"team_id": "...", "incident_id": "...", "reason": "short"}]}"""

ALLOC_INSTR = """You are the Resource Allocation Agent, the referee between Dispatch and the hospitals.
You get Dispatch's proposals (some ambulances may be wanted by several incidents), incident
priorities and waiting victims, and hospital capacity, specialties and travel minutes.
Decide: each ambulance goes to at most ONE incident; urgent incidents win clashes; carry at most
2 victims and no more than are waiting; while serious victims wait anywhere, only count serious ones; choose a hospital with capacity (trauma specialty for
loads with immediate victims). If some hospitals rejected or countered earlier requests, they are
listed in "feedback": respect them.
Tool: hospital_options(incident_id, victims) lists hospitals that can take the load, nearest first.
Reply with ONLY JSON: {"decisions": [{"ambulance_id": "...", "incident_id": "...",
"victim_count": 2, "hospital_id": "...", "reason": "short"}]}"""

HOSP_INSTR = """You are the Hospital Coordination Agent for all hospitals. Your goal: admit patients
safely without overloading any ER, and keep trauma-center capacity for critical patients.
You get admission requests (request_id, hospital, victims, whether any is immediate) and each
hospital's free capacity (beds AND staff), kind and specialties, plus how many critical victims
are still waiting without a unit. For each request answer ACCEPT, REJECT (no capacity), or
COUNTER with alt_hospital_id (e.g. a non-critical load would use the trauma center's last beds).
Tool: check_capacity(hospital_id, victims) says whether a request fits. Call it for every request
and never REJECT a request it says fits, unless you COUNTER to keep trauma beds for critical patients.
Call it only ONCE per request, then write your reply. Always give a short text "reason".
Each request also has "fits": YES means it fits; never REJECT a request whose fits is YES.
Reply with ONLY JSON: {"responses": [{"request_id": "...", "decision": "ACCEPT",
"alt_hospital_id": null, "reason": "short"}]}"""


STAFF_INSTR = """You are the Hospital Coordination Agent, now speaking for a DONOR hospital that has been
asked to lend one doctor and one nurse to a full hospital, so it can open overflow (surge) beds.
You get the donor's free capacity, beds in use, and the critical victims waiting in the city.
OFFER if the donor can spare them without leaving its own patients short (the simulator has already
checked that its remaining staff can still cover all its beds); DECLINE if the donor is itself busy
and likely to need them soon. Tool: check_capacity(hospital_id, victims) gives live capacity.
Reply with ONLY JSON: {"decision": "OFFER", "reason": "short"}"""
STAFF_COOLDOWN_MIN = 15      


class AgentTeam:
    def __init__(self, llm, mode="rule"):
        self.llm, self.mode = llm, mode
        self.board = {}         
        self.messages = []
        self.stats = dict(decision_rounds=0, proposals=0, contested_ambulances=0,
                          admission_requests=0, accepts=0, rejects=0, counters=0, unresolved=0,
                          staff_requests=0, staff_offers=0, staff_declines=0, llm_fallbacks=0)
        self._last_staff_request = {}  
        self._last_signature = None
        self._counters = {}       

    def msg(self, sim, sender, recipient, msg_type, incident_id, action, reason="", by="rule"):
        self.messages.append(Message(t=sim.t, sender=sender, recipient=recipient, msg_type=msg_type,
                                     incident_id=incident_id, proposed_action=action, reason=reason,
                                     decided_by=by).model_dump())

    def ask(self, name, instr, payload, schema, sim=None, tools=(), used=None):
        if self.mode != "llm":
            return None, "rule"
        out = self.llm.ask(name, instr, payload, schema, tools=tools, sim=sim, team=self, used=used)
        if out is None:
            self.stats["llm_fallbacks"] += 1
            return None, "rule"
        return out, "llm"

    @staticmethod
    def triage_order(victims):
        return sorted(victims, key=lambda v: (SEV[v["severity"]], v["victim_id"]))

    def assess_rule(self, inc):
        r = inc.get("report", "").lower()
        num = lambda pat: sum(int(n) for n in re.findall(r"(\d+)\s+(?:people\s+)?" + pat, r))
        imm = num(r"(?:unresponsive|not breathing|critical|trapped|with severe)")
        dl = num(r"(?:with serious|seriously|with broken|with fractures|with burns)")
        mn = num(r"(?:walking wounded|with minor|minor)")
        if imm + dl + mn == 0:  # free text with no counts: use keywords
            total = int((re.findall(r"(\d+)\s+(?:people|victims|injured)", r) or ["1"])[0])
            imm = 1 if any(k in r for k in ("unresponsive", "not breathing", "trapped", "severe", "critical")) else 0
            dl = 1 if any(k in r for k in ("broken", "fracture", "burn", "serious")) else 0
            mn = max(0, total - imm - dl)
        pr = 1 if imm else 2 if dl else 3
        return S.Assess(incident_id=inc["incident_id"], priority=pr, immediate=imm, delayed=dl,
                        minor=mn, reason="keywords in report")

    def assessment_step(self, sim):
        new = [i for i in sim.active_incidents() if i["incident_id"] not in self.board]
        if not new:
            return
        payload = {"incidents": [{"incident_id": i["incident_id"], "report": i.get("report", "")} for i in new]}
        out, by = self.ask("Incident Assessment Agent", ASSESS_INSTR, payload, S.AssessmentOut,
                           sim=sim, tools=T.ASSESS_TOOLS)
        got = {a.incident_id: a for a in (out.assessments if out else [])}
        for inc in new:
            a, who = (got[inc["incident_id"]], by) if inc["incident_id"] in got else (self.assess_rule(inc), "rule")
            self.board[inc["incident_id"]] = a.model_dump() | {"arrival_min": inc["arrival_min"]}
            self.msg(sim, "Assessment", "Dispatch", "PRIORITY", inc["incident_id"],
                     f"priority {a.priority}: {a.immediate} immediate, {a.delayed} delayed, {a.minor} minor",
                     a.reason, who)

    def view(self, sim):
        free_amb = [a for a in sim.ambulances.values() if a["status"] == "available"]
        incidents = []
        for inc in sim.active_incidents():
            b = self.board[inc["incident_id"]]
            waiting = sim.waiting_victims(inc["incident_id"])
            visited = inc["first_unit_min"] is not None
            if visited:  
                serious = sum(v["severity"] != "minor" for v in waiting)
            else:         
                gone = len(inc["victims"]) - len(waiting)
                serious = min(len(waiting), max(0, b["immediate"] + b["delayed"] - gone))
            incidents.append({"incident_id": inc["incident_id"], "priority": b["priority"],
                              "waiting": len(waiting), "serious_waiting": serious,
                              "minor_waiting": len(waiting) - serious, "triage_known": visited,
                              "minutes_waiting": sim.t - inc["arrival_min"],
                              "arrival_min": inc["arrival_min"],
                              "needs_team": inc["needs_responders"][0] if inc["needs_responders"] else None,
                              "team_assigned": inc["team_id"]})
        ambs = [{"ambulance_id": a["ambulance_id"], "level": a["level"],
                 "eta": {i["incident_id"]: round(sim.estimate_min(a["location"], sim.incidents[i["incident_id"]]["location"]))
                         for i in incidents}} for a in free_amb]
        teams = [{"team_id": t["team_id"], "team_type": t["team_type"]} for t in sim.teams.values()
                 if t["status"] == "available"]
        return incidents, ambs, teams

    @staticmethod
    def reachable(inc):
        return not (inc["needs_team"] and not inc["team_assigned"])

    def demand(self, inc, incidents):
        serious_anywhere = any(i["serious_waiting"] > 0 and self.reachable(i) for i in incidents)
        return inc["serious_waiting"] if serious_anywhere else inc["waiting"]

    @staticmethod
    def order(incidents):
        return sorted(incidents, key=lambda i: (i["priority"], -i["serious_waiting"], i["arrival_min"]))

    def hospital_view(self, sim, incident_id=None, used=None):
        used = used or {}
        out = []
        for h in sim.hospitals.values():
            row = {"hospital_id": h["hospital_id"], "kind": h["kind"], "specialties": h["specialties"],
                   "free_capacity": sim.hospital_capacity(h["hospital_id"]) - used.get(h["hospital_id"], 0)}
            if incident_id:
                row["travel_min"] = round(sim.estimate_min(sim.incidents[incident_id]["location"], h["location"]))
            out.append(row)
        return out

    def dispatch_rule(self, incidents, ambs, teams):
        amb_props, team_props, teams_left = [], [], list(teams)
        for inc in self.order(incidents):
            iid = inc["incident_id"]
            if inc["needs_team"] and not inc["team_assigned"]:
                match = [t for t in teams_left if t["team_type"] == inc["needs_team"]]
                if match:
                    teams_left.remove(match[0])
                    team_props.append(S.TeamProposal(team_id=match[0]["team_id"], incident_id=iid,
                                                     reason=f"{inc['needs_team']} needed before loading"))
                continue         
            score = lambda a: a["eta"][iid] - (5 if inc["priority"] == 1 and a["level"] == "ALS" else 0)
            for a in sorted(ambs, key=score)[: math.ceil(self.demand(inc, incidents) / 2)]:
                amb_props.append(S.AmbProposal(ambulance_id=a["ambulance_id"], incident_id=iid,
                                               reason=f"eta {a['eta'][iid]} min, priority {inc['priority']}, "
                                                      f"{inc['serious_waiting']} serious waiting"))
        return S.DispatchOut(ambulance_proposals=amb_props, team_proposals=team_props)

    def allocation_rule(self, sim, incidents, ambs, proposals, feedback, base_used=None):
        by_inc = {}
        for p in proposals:
            by_inc.setdefault(p.incident_id, []).append(p.ambulance_id)
        claimed, decisions, used = set(), [], dict(base_used or {})
        taken = set()
        for inc in self.order(incidents):
            iid, left = inc["incident_id"], self.demand(inc, incidents)
            wanted = by_inc.get(iid, [])
            spare = sorted(ambs, key=lambda a: a["eta"][iid])
            for amb_id in wanted:
                if left <= 0:
                    break
                if amb_id in claimed:       
                    alt = next((a["ambulance_id"] for a in spare if a["ambulance_id"] not in claimed), None)
                    if not alt:
                        break
                    amb_id = alt
                claimed.add(amb_id)
                count = min(2, left)
                left -= count
                decisions.append([amb_id, iid, count])
        out = []
        for amb_id, iid, count in decisions:
            load = self.take(sim, iid, count, taken)
            hosp = self.pick_hospital(sim, iid, load, used, feedback)
            if hosp:
                used[hosp] = used.get(hosp, 0) + count
                out.append(S.Allocation(ambulance_id=amb_id, incident_id=iid, victim_count=count,
                                        hospital_id=hosp, reason="urgency first; nearest suitable hospital"))
        return S.AllocationOut(decisions=out)

    def take(self, sim, iid, count, taken):
        """Pick the most critical waiting victims at an incident that nobody has claimed yet."""
        load = [v for v in self.triage_order(sim.waiting_victims(iid)) if v["victim_id"] not in taken][:count]
        taken.update(v["victim_id"] for v in load)
        return load

    def pick_hospital(self, sim, iid, load, used, feedback):
        count = len(load)
        critical = any(v["severity"] == "immediate" for v in load)
        banned = {f["hospital_id"] for f in feedback if f["incident_id"] == iid and f["decision"] == "REJECT"}
        options = [h for h in self.hospital_view(sim, iid, used) if h["free_capacity"] >= count
                   and h["hospital_id"] not in banned]
        suggested = self._counters.get(iid)
        if suggested and any(h["hospital_id"] == suggested for h in options):
            return suggested                      
        if critical and any("trauma" in h["specialties"] for h in options):
            options = [h for h in options if "trauma" in h["specialties"]]
        return min(options, key=lambda h: h["travel_min"])["hospital_id"] if options else None

    def hospital_rule(self, sim, requests, critical_waiting, used=None):
        remaining = {h["hospital_id"]: h["free_capacity"] for h in self.hospital_view(sim, used=used)}
        reserve = min(2, critical_waiting)
        out = []
        for r in requests:
            h, n = r["hospital_id"], r["victims"]
            others = sorted([x for x in self.hospital_view(sim, r["incident_id"])
                             if x["hospital_id"] != h and remaining[x["hospital_id"]] >= n],
                            key=lambda x: x["travel_min"])
            alt = others[0]["hospital_id"] if others else None
            if remaining[h] < n:
                out.append(S.HospitalResponse(request_id=r["request_id"], decision="COUNTER" if alt else "REJECT",
                                              alt_hospital_id=alt, reason=f"{h} full"))
            elif (sim.hospitals[h]["kind"] == "trauma center" and not r["has_immediate"]
                  and remaining[h] - n < reserve and alt):
                out.append(S.HospitalResponse(request_id=r["request_id"], decision="COUNTER", alt_hospital_id=alt,
                                              reason=f"{h} holding {reserve} bed(s) for critical patients"))
            else:
                remaining[h] -= n
                out.append(S.HospitalResponse(request_id=r["request_id"], decision="ACCEPT", reason="capacity ok"))
        return S.HospitalOut(responses=out)

    def staff_rule(self, sim, donor, critical_waiting):
        h = sim.hospitals[donor]
        in_use = h["beds_total"] - h["beds_available"]
        if in_use <= h["beds_total"] // 2 or critical_waiting == 0:
            return S.StaffDecision(decision="OFFER", reason=f"{donor} is only {in_use}/{h['beds_total']} full")
        return S.StaffDecision(decision="DECLINE", reason=f"{donor} is busy ({in_use}/{h['beds_total']} beds used)")

    def staff_sharing_step(self, sim):
        """A full hospital asks a less busy one for a doctor + nurse to open surge beds."""
        critical_waiting = sum(v["severity"] == "immediate" for v in sim.waiting_victims())
        if critical_waiting == 0:
            return []
        actions = []
        for h in sorted(sim.hospitals.values(), key=lambda x: x["kind"] != "trauma center"):
            hid = h["hospital_id"]
            if (sim.hospital_capacity(hid) > 0 or h.get("diverted") or sim.surge_left(hid) == 0
                    or sim.t - self._last_staff_request.get(hid, -999) < STAFF_COOLDOWN_MIN):
                continue
            donors = sorted([d for d in sim.hospitals if d != hid and sim.can_lend(d, hid)],
                            key=lambda d: sim.hospitals[d]["beds_available"], reverse=True)
            if not donors:
                continue
            donor = donors[0]
            self._last_staff_request[hid] = sim.t
            self.stats["staff_requests"] += 1
            self.msg(sim, hid, donor, "STAFF_REQUEST", None, f"{donor} lends 1 doctor + 1 nurse to {hid}",
                     f"{hid} is full; {critical_waiting} critical victims waiting")
            dh = sim.hospitals[donor]
            payload = {"minute": sim.t, "donor": donor, "requesting_hospital": hid,
                       "donor_free_capacity": sim.hospital_capacity(donor),
                       "donor_beds_in_use": dh["beds_total"] - dh["beds_available"],
                       "donor_beds_total": dh["beds_total"], "critical_victims_waiting": critical_waiting}
            out, by = self.ask("Staff Sharing (Hospital Coordination)", STAFF_INSTR, payload, S.StaffDecision,
                               sim=sim, tools=T.HOSP_TOOLS)
            if out is None:
                out, by = self.staff_rule(sim, donor, critical_waiting), "rule"
            if out.decision == "OFFER":
                self.stats["staff_offers"] += 1
                self.msg(sim, donor, hid, "STAFF_OFFER", None, f"1 doctor + 1 nurse to {hid}", out.reason, by)
                actions.append(LendStaff(donor, hid, out.reason))
                self.msg(sim, "Allocation", "all", "DECISION", None,
                         f"staff loan {donor} -> {hid} approved (opens surge beds)", "critical demand")
            else:
                self.stats["staff_declines"] += 1
                self.msg(sim, donor, hid, "STAFF_DECLINE", None, f"no staff for {hid}", out.reason, by)
        return actions

    def clean_allocations(self, sim, out, ambs):
        free = {a["ambulance_id"] for a in ambs}
        left = {i["incident_id"]: len(sim.waiting_victims(i["incident_id"])) for i in sim.active_incidents()}
        seen, good = set(), []
        for d in out.decisions:
            if (d.ambulance_id in free and d.ambulance_id not in seen and d.incident_id in left
                    and d.hospital_id in sim.hospitals and left[d.incident_id] > 0):
                d.victim_count = min(d.victim_count, left[d.incident_id])
                left[d.incident_id] -= d.victim_count
                seen.add(d.ambulance_id)
                good.append(d)
        return good

    def __call__(self, sim):
        self.assessment_step(sim)
        staff_actions = self.staff_sharing_step(sim)
        incidents, ambs, teams = self.view(sim)
        need_team = any(i["needs_team"] and not i["team_assigned"] for i in incidents)
        if not incidents or (not ambs and not (need_team and teams)):
            return staff_actions
        signature = (tuple(sorted(a["ambulance_id"] for a in ambs)),
                     tuple((i["incident_id"], i["waiting"], i["team_assigned"]) for i in incidents),
                     tuple(h["free_capacity"] for h in self.hospital_view(sim)))
        if signature == self._last_signature:      
            return staff_actions
        self.stats["decision_rounds"] += 1
        actions = list(staff_actions)

        payload = {"minute": sim.t, "incidents": incidents, "free_ambulances": ambs, "free_teams": teams}
        out, by = self.ask("Dispatch Agent", DISPATCH_INSTR, payload, S.DispatchOut,
                           sim=sim, tools=T.DISPATCH_TOOLS)
        if out is None or (ambs and not out.ambulance_proposals):
            out, by = self.dispatch_rule(incidents, ambs, teams), "rule"
        valid_amb = {a["ambulance_id"] for a in ambs}
        valid_inc = {i["incident_id"] for i in incidents}
        props = [p for p in out.ambulance_proposals if p.ambulance_id in valid_amb and p.incident_id in valid_inc]
        for p in props:
            self.msg(sim, "Dispatch", "Allocation", "PROPOSAL", p.incident_id,
                     f"{p.ambulance_id} -> {p.incident_id}", p.reason, by)
        self.stats["proposals"] += len(props)

        used_teams = set()
        for tp in out.team_proposals:
            team = sim.teams.get(tp.team_id)
            inc = sim.incidents.get(tp.incident_id)
            if (team and inc and team["status"] == "available" and tp.team_id not in used_teams
                    and not inc["team_id"] and team["team_type"] in inc["needs_responders"]):
                used_teams.add(tp.team_id)
                actions.append(AssignResponder(tp.team_id, tp.incident_id, tp.reason))
                self.msg(sim, "Allocation", "Dispatch", "DECISION", tp.incident_id,
                         f"{tp.team_id} -> {tp.incident_id}", tp.reason, by)
        wanted = {}
        for p in props:
            wanted.setdefault(p.ambulance_id, []).append(p.incident_id)
        for amb_id, incs in wanted.items():
            if len(set(incs)) > 1:
                self.stats["contested_ambulances"] += 1
                self.msg(sim, "Allocation", "Dispatch", "CONFLICT", None,
                         f"{amb_id} wanted by {', '.join(sorted(set(incs)))}", "same ambulance proposed twice")

        feedback, final, taken = [], [], set()
        self._counters = {}
        critical_waiting = sum(v["severity"] == "immediate" for v in sim.waiting_victims())
        pending_amb = ambs
        for rnd in range(1, MAX_ROUNDS + 1):
            payload = {"minute": sim.t, "round": rnd, "proposals": [p.model_dump() for p in props],
                       "incidents": incidents, "free_ambulances": pending_amb,
                       "hospitals": self.hospital_view(sim, used={k: v for k, v in _used(final).items()}),
                       "feedback": feedback}
            out, by = self.ask("Resource Allocation Agent", ALLOC_INSTR, payload, S.AllocationOut,
                               sim=sim, tools=T.ALLOC_TOOLS, used=_used(final))
            decisions = self.clean_allocations(sim, out, pending_amb) if out else None
            if not decisions:
                decisions = self.allocation_rule(sim, incidents, pending_amb, props, feedback,
                                                 base_used=_used(final)).decisions
                by = "rule"
            decisions = [d for d in decisions if all(d.ambulance_id != f.ambulance_id for f in final)]
            if not decisions:
                break

            requests = []
            round_taken = set(taken)
            for k, d in enumerate(decisions):
                load = self.take(sim, d.incident_id, d.victim_count, round_taken)
                d.victim_count = max(1, len(load))
                rid = f"R{rnd}-{k}"
                requests.append({"request_id": rid, "incident_id": d.incident_id, "hospital_id": d.hospital_id,
                                 "victims": d.victim_count,
                                 "has_immediate": any(v["severity"] == "immediate" for v in load),
                                 "_decision": d})
                self.msg(sim, "Allocation", d.hospital_id, "ADMISSION_REQUEST", d.incident_id,
                         f"{d.ambulance_id} -> {d.incident_id} -> {d.hospital_id} ({d.victim_count} victims)",
                         d.reason, by)
            self.stats["admission_requests"] += len(requests)

            public = [{k: v for k, v in r.items() if k != "_decision"} for r in requests]
            remaining = {h["hospital_id"]: h["free_capacity"] for h in self.hospital_view(sim, used=_used(final))}
            for r in public:            
                fits = remaining[r["hospital_id"]] >= r["victims"]
                r["fits"] = "YES" if fits else "NO"
                remaining[r["hospital_id"]] -= r["victims"] if fits else 0
            payload = {"minute": sim.t, "requests": public, "hospitals": self.hospital_view(sim, used=_used(final)),
                       "critical_victims_waiting": critical_waiting}
            out, hby = self.ask("Hospital Coordination Agent", HOSP_INSTR, payload, S.HospitalOut,
                                sim=sim, tools=T.HOSP_TOOLS, used=_used(final))
            rule = {r.request_id: r for r in self.hospital_rule(sim, public, critical_waiting, used=_used(final)).responses}
            answers = {r.request_id: r for r in (out.responses if out else []) if r.request_id in rule}

            feedback = []
            for r in requests:
                ans, who = (answers[r["request_id"]], hby) if r["request_id"] in answers else (rule[r["request_id"]], "rule")
                d = r["_decision"]
                self.msg(sim, d.hospital_id, "Allocation", ans.decision, d.incident_id,
                         f"{d.ambulance_id} -> {d.hospital_id}" + (f" (try {ans.alt_hospital_id})" if ans.alt_hospital_id else ""),
                         ans.reason, who)
                if ans.decision == "ACCEPT":
                    self.stats["accepts"] += 1
                    final.append(d)
                    self.take(sim, d.incident_id, d.victim_count, taken)
                else:
                    self.stats["rejects" if ans.decision == "REJECT" else "counters"] += 1
                    feedback.append({"incident_id": d.incident_id, "hospital_id": d.hospital_id,
                                     "decision": ans.decision, "alt_hospital_id": ans.alt_hospital_id,
                                     "reason": ans.reason})
            if not feedback:
                break
            pending_amb = [a for a in ambs if all(a["ambulance_id"] != f.ambulance_id for f in final)]
            props = [p for p in props if p.ambulance_id in {a["ambulance_id"] for a in pending_amb}]
            self._counters = {f["incident_id"]: f["alt_hospital_id"] for f in feedback
                              if f["decision"] == "COUNTER" and f["alt_hospital_id"] in sim.hospitals}
        else:
            if feedback:
                self.stats["unresolved"] += len(feedback)
                for f in feedback:
                    self.msg(sim, "Allocation", "Dispatch", "UNRESOLVED", f["incident_id"],
                             f"no agreement after {MAX_ROUNDS} rounds", f["reason"])
        loaded = set()
        for d in final:
            load = self.take(sim, d.incident_id, d.victim_count, loaded)
            if load:
                actions.append(Dispatch(d.ambulance_id, d.incident_id, [v["victim_id"] for v in load],
                                        d.hospital_id, d.reason))
                self.msg(sim, "Allocation", "all", "DECISION", d.incident_id,
                         f"{d.ambulance_id} -> {d.incident_id} -> {d.hospital_id} ({len(load)} victims)", d.reason)
        self._last_signature = None if len(actions) > len(staff_actions) else signature
        return actions


def _used(final):
    used = {}
    for d in final:
        used[d.hospital_id] = used.get(d.hospital_id, 0) + d.victim_count
    return used