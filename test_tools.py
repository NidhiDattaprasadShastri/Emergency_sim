import json
from pathlib import Path

from sim.engine import Simulation, load_world
from team import schemas as S
from team import tools as T
from team.llm import LLM
from team.team import HOSP_INSTR, AgentTeam

cfg = json.loads(Path("config.json").read_text())["ollama"]
sim = Simulation(load_world("data/world_v1.0_seed42.json"))
team = AgentTeam(LLM(enabled=False), mode="rule")
for _ in range(60):
    sim.advance()
    sim.apply(team(sim))

inc = sim.active_incidents()[0]["incident_id"]
payload = {"minute": sim.t,
           "requests": [{"request_id": "R1-0", "incident_id": inc, "hospital_id": "H-B", "victims": 2,
                         "has_immediate": True}],
           "hospitals": team.hospital_view(sim),
           "critical_victims_waiting": sum(v["severity"] == "immediate" for v in sim.waiting_victims())}

llm = LLM(enabled=True, base_url=cfg["base_url"], model=cfg["model"])
out = llm.ask("Hospital Coordination Agent", HOSP_INSTR, payload, S.HospitalOut,
              tools=T.HOSP_TOOLS, sim=sim, team=team)
print("\nTools called:", llm.tool_calls or "none")
print("Agent answer:", out.model_dump() if out else f"invalid reply ({llm.last_error})")