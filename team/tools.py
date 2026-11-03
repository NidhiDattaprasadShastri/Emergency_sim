from dataclasses import dataclass, field

from agents import RunContextWrapper, function_tool


@dataclass
class ToolContext:
    sim: object
    team: object
    used: dict = field(default_factory=dict)    
    calls: dict = field(default_factory=dict)     


def _count(ctx, name):
    ctx.context.calls[name] = ctx.context.calls.get(name, 0) + 1
@function_tool
def triage_keywords(ctx: RunContextWrapper[ToolContext], incident_id: str) -> str:
    _count(ctx, "triage_keywords")
    inc = ctx.context.sim.incidents.get(incident_id)
    if not inc:
        return f"unknown incident {incident_id}"
    a = ctx.context.team.assess_rule(inc)
    return (f"{incident_id} keyword estimate: priority {a.priority}, {a.immediate} immediate, "
            f"{a.delayed} delayed, {a.minor} minor")

@function_tool
def travel_time(ctx: RunContextWrapper[ToolContext], unit_id: str, incident_id: str) -> str:
    _count(ctx, "travel_time")
    sim = ctx.context.sim
    unit = sim.ambulances.get(unit_id) or sim.teams.get(unit_id)
    inc = sim.incidents.get(incident_id)
    if not unit or not inc:
        return "unknown unit or incident"
    minutes = round(sim.estimate_min(unit["location"], inc["location"]))
    return f"{unit_id} ({unit['status']}) to {incident_id}: about {minutes} min"


@function_tool
def incident_status(ctx: RunContextWrapper[ToolContext], incident_id: str) -> str:
    _count(ctx, "incident_status")
    sim = ctx.context.sim
    inc = sim.incidents.get(incident_id)
    if not inc:
        return f"unknown incident {incident_id}"
    waiting = len(sim.waiting_victims(incident_id))
    need = inc["needs_responders"][0] if inc["needs_responders"] else "none"
    return (f"{incident_id}: {waiting} waiting, team needed: {need}, team assigned: {inc['team_id'] or 'no'}, "
            f"scene safe: {'yes' if inc['responder_on_scene'] or need == 'none' else 'no'}")

@function_tool
def hospital_options(ctx: RunContextWrapper[ToolContext], incident_id: str, victims: int) -> str:
    _count(ctx, "hospital_options")
    sim, team = ctx.context.sim, ctx.context.team
    if incident_id not in sim.incidents:
        return f"unknown incident {incident_id}"
    rows = [h for h in team.hospital_view(sim, incident_id, ctx.context.used) if h["free_capacity"] >= victims]
    if not rows:
        return f"no hospital can take {victims} victims right now"
    rows.sort(key=lambda h: h["travel_min"])
    return "; ".join(f"{h['hospital_id']} ({h['kind']}, trauma: {'yes' if 'trauma' in h['specialties'] else 'no'}, "
                     f"free {h['free_capacity']}, {h['travel_min']} min)" for h in rows)

@function_tool
def check_capacity(ctx: RunContextWrapper[ToolContext], hospital_id: str, victims: int) -> str:
    _count(ctx, "check_capacity")
    sim = ctx.context.sim
    h = sim.hospitals.get(hospital_id)
    if not h:
        return f"unknown hospital {hospital_id}"
    free = sim.hospital_capacity(hospital_id) - ctx.context.used.get(hospital_id, 0)
    critical_waiting = sum(v["severity"] == "immediate" for v in sim.waiting_victims())
    fits = "YES" if free >= victims else "NO"
    note = ""
    if h["kind"] == "trauma center":
        note = f" Trauma center: {critical_waiting} critical victims still waiting elsewhere."
    return f"{hospital_id}: free capacity {free}. {victims} victim(s) fit: {fits}.{note}"


ASSESS_TOOLS = [triage_keywords]
DISPATCH_TOOLS = [travel_time, incident_status]
ALLOC_TOOLS = [hospital_options]
HOSP_TOOLS = [check_capacity]