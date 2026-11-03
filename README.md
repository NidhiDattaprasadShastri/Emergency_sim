# Multi-Agent Emergency Response Coordination

A multi-agent system, built with the **OpenAI Agents SDK for Python**, that coordinates
ambulances, responder teams, hospital ER beds, doctors and nurses during a multi-incident
emergency. Four agents with different goals and their own SDK tools communicate through
structured messages and negotiate when their priorities conflict: over ambulances, over
hospital beds, and over medical staff. A reproducible simulation shows that this
coordination **measurably improves outcomes** compared with a greedy, non-negotiating baseline.

---

## 1. Quick start (reproduce everything)

```bash
python -m pip install -r requirements.txt

ollama pull qwen2.5:3b                   # free local model used by the agents (https://ollama.com)
python test_ollama.py                    # checks the Agents SDK <-> Ollama connection
python test_tools.py                     # checks that an agent really calls its SDK tool

python generate_world.py                 # 1. world for seed 42 with LLM-written incident reports
python run_baseline.py                   # 2. greedy baseline
python run_agents.py --mode rule         # 3. four agents with their rule-based policies (instant)
python run_experiments.py                # 4. 3 scenarios x 10 seeds, baseline vs agents (~1 min)
python run_agents.py --mode llm          # 5. four agents thinking with the LLM (~90 min on a laptop CPU)
python -m streamlit run app.py           # 6. interactive dashboard (http://localhost:8501)
```

`python generate_world.py --no-llm` builds the same world with template reports instead.

**Reproducibility.** Every random choice comes from one seed. For seed 42:

| Command | Fingerprint (same on every run) |
|---|---|
| `python generate_world.py --no-llm` | `df95d7df8678` |
| `python generate_world.py` (LLM reports, cached) | `c69214bbde81` |

LLM-written reports are cached in `cache/reports_seed42_llm.json`, so the LLM writes each report
only once and reruns are identical. Agents run with temperature 0, and every run saves its full
event log and message log in `runs/`.

---

## 2. Project structure

| File | Purpose |
|---|---|
| `config.json` | All settings: seed, counts, probabilities, hospital profiles, Ollama model |
| `generate_world.py` | Seeded synthetic world → `data/world_v1.0_seed42.json` |
| `models.py` | Pydantic schemas that validate every generated record |
| `narrate.py` | Report Writer agent (Agents SDK + Ollama) turns ground truth into 911-style reports, cached |
| `sim/engine.py` | Minute-by-minute simulation engine; checks every action |
| `sim/baseline.py` | Greedy baseline policy (no priorities, no negotiation) |
| `team/team.py` | The four agents, their instructions and decision policies, and the negotiation protocol |
| `team/tools.py` | The agents' SDK tools (`@function_tool`) |
| `team/llm.py` | Connects the Agents SDK to the local Ollama model |
| `team/schemas.py` | The JSON format each agent must reply with (Pydantic) |
| `team/messages.py` | The structured message format |
| `run_baseline.py`, `run_agents.py` | Run one world and save `runs/*.json` |
| `run_experiments.py` | Experiments → `results/experiments.csv`, `results/summary.csv` |
| `dashboard_data.py`, `app.py` | Streamlit + Plotly dashboard |
| `test_ollama.py`, `test_tools.py` | Quick checks of the LLM connection and tool calling |

---

## 3. Simulated environment

| Entity | Count | Attributes |
|---|---|---|
| Incidents | 12–15 | location, 1–8 victims, START triage per victim, deterioration deadline, required team, free-text report |
| Hospitals | 3 | location, ER beds (8 / 6 / 4 = 18), available beds, specialties, staff, surge beds |
| Ambulances | 5 | 2 ALS + 3 BLS, capacity 2 patients, location, status, travel time |
| Medical staff | 26 | 10 doctors + 16 nurses, skills, availability, workload caps (doctor 4, nurse 6) |
| Responder teams | 3 | fire, rescue, hazmat; location, status |

Seed 42 world: 12 incidents, 50 victims (11 immediate, 13 delayed, 26 minor) competing for
18 ER beds. Resources are deliberately scarce, so agents must make trade-offs.

### Data generation

1. **Seed:** one master seed drives a NumPy random generator for all entities (locations,
   arrival times, victims, triage, staff skills).
2. **Narrate:** the code creates the ground truth; an LLM agent (OpenAI Agents SDK + Ollama)
   turns it into a free-text 911-style report, e.g. *"Both victims are experiencing minor skin
   irritation and have minor burns. They are conscious and able to provide basic information."*
   The Incident Assessment Agent reads this text, never the ground truth.
3. **Validate:** Pydantic checks every record (triage levels, locations inside the grid,
   capacities, ID consistency).
4. **Store:** versioned JSON; LLM reports cached.

### Assumptions

- City: 20 × 20 km schematic grid. Travel time = Manhattan distance ÷ 40 km/h × random
  traffic factor (0.8–1.5).
- Incidents arrive at random times (Poisson process, 7 per hour over 120 minutes), kept to 12–15.
- Victim severity (START triage): 20% immediate, 30% delayed, 50% minor.
- Immediate victims deteriorate if not in hospital within 60 minutes of the incident.
- Incident types and the team they need first: vehicle collision and structural collapse →
  rescue; building fire → fire; chemical leak → hazmat; medical emergency → none.
- A responder team needs 10 minutes on scene to make it safe; ambulances wait until then.
- Loading takes 3 minutes. Treatment occupies a bed, one doctor and one nurse for
  90 / 60 / 30 minutes (immediate / delayed / minor). Immediate patients get a
  trauma-skilled doctor when one is free.
- A hospital's capacity is the minimum of free beds, free doctor slots and free nurse slots.
- Beds and staff are reserved at dispatch time, so two ambulances cannot claim the same bed.
- Victims' true triage becomes known once the first unit reaches the scene; before that,
  agents rely on the Incident Assessment Agent's estimate. Crews load the most critical first.
- **Staff sharing:** each hospital has unstaffed surge beds (H-A 2, H-B 2, H-C 1). A loan of one
  idle doctor + one idle nurse takes 15 minutes to arrive and opens up to 2 surge beds. A hospital
  may lend only if its remaining staff still covers all its own beds. Loans last until the end of
  the run. A hospital waits 15 minutes before asking again.
- Hospital diversion (scenario): the hospital accepts no new patients from that minute.
- The simulation runs for 240 minutes (arrivals stop at 120).

---

## 4. The agents

| Agent | Objective | SDK tools | Decides |
|---|---|---|---|
| **Incident Assessment** | Estimate urgency correctly from free text | `triage_keywords` | priority 1–3 and triage counts |
| **Dispatch** | Get help to the most urgent victims fast | `travel_time`, `incident_status` | ambulance and team proposals (may clash) |
| **Resource Allocation** | Referee: resolve competing requests fairly | `hospital_options` | one incident per ambulance, victims per trip, hospital; approves staff loans |
| **Hospital Coordination** | Avoid overload; keep trauma beds for critical patients; share staff | `check_capacity` | ACCEPT / REJECT / COUNTER admissions; OFFER / DECLINE staff loans |

Each agent has two "brains" with the same inputs and outputs:
- **LLM mode:** an Agents SDK `Agent` with its own instructions and tools, running on a local
  Ollama model (`qwen2.5:3b`, OpenAI-compatible API), temperature 0, replying in JSON validated
  by Pydantic. Tools read live facts from the simulation through the SDK run context.
- **Rule mode:** a deterministic decision policy, used for the 60-run experiments and as the
  fallback whenever an LLM reply is invalid.

The engine validates every final action (ambulance free, hospital has capacity, victims waiting,
donor can spare staff), so no agent can break the simulation's rules.

### Structured messages

Every message has `t`, `sender`, `recipient`, `msg_type`, `incident_id`, `proposed_action`,
`reason` and `decided_by` (llm or rule). Types: PRIORITY, PROPOSAL, CONFLICT,
ADMISSION_REQUEST, ACCEPT, REJECT, COUNTER, DECISION, UNRESOLVED, STAFF_REQUEST,
STAFF_OFFER, STAFF_DECLINE.

### Negotiation protocol (one decision round)

1. Assessment → Dispatch: **PRIORITY** for each new incident.
2. Staff sharing: a full hospital → another hospital: **STAFF_REQUEST**; the other hospital
   answers **STAFF_OFFER** or **STAFF_DECLINE**; Allocation approves the loan (**DECISION**).
3. Dispatch → Allocation: **PROPOSAL**s. Dispatch advocates for each incident, so the same
   ambulance can be proposed for several incidents; each clash is logged as a **CONFLICT**.
4. Allocation → Hospital: **ADMISSION_REQUEST**s. The more urgent incident wins each clash;
   the loser gets the next-best free ambulance.
5. Hospital → Allocation: **ACCEPT**, **REJECT** or **COUNTER** with an alternative hospital
   (for example, a non-critical load that would use the trauma center's last beds).
6. Steps 4–5 repeat for up to **3 rounds**; counter-offers are honoured in the next round.
   Anything still unresolved is logged as **UNRESOLVED**.
7. Allocation → all: final **DECISION**s become engine actions.

A round runs only when something has changed (new incident, freed ambulance, freed bed).

---

## 5. Results

### Experiments: 3 scenarios × 10 seeds (baseline vs. agents, rule-mode policies)

- **normal:** the world as generated.
- **surge:** 6 incidents squeezed into 15 minutes (mass-casualty surge).
- **diversion:** the trauma center (H-A) stops taking new patients at minute 60.

| Scenario | Metric | Baseline | Agents | Agents better in |
|---|---|---|---|---|
| normal | critical victims missed | 8.7 | **6.8** | 8/10 seeds |
| normal | victims served | 29.1 | **34.6** | 7/10 |
| normal | avg response time (min) | 63.9 | **47.8** | 9/10 |
| normal | critical victim to hospital (min) | 90.6 | **73.7** | 8/10 |
| surge | critical victims missed | 8.6 | **6.6** | 9/10 |
| surge | victims served | 27.2 | **32.8** | 8/10 |
| surge | avg response time (min) | 68.1 | **50.1** | 8/10 |
| surge | critical victim to hospital (min) | 97.3 | **76.1** | 7/10 |
| diversion | critical victims missed | 8.7 | **7.2** | 7/10 |
| diversion | victims served | 21.6 | **25.5** | 6/10 |
| diversion | avg response time (min) | 55.0 | **47.2** | 7/10 |
| diversion | critical victim to hospital (min) | 90.2 | **84.9** | 4/10 |

Coordination helps most in the **surge** scenario: 2 fewer critical victims missed and critical
patients in hospital about 21 minutes sooner. Adding staff sharing between hospitals raised the
gain in victims served from +3.1 / +4.3 / +2.5 to **+5.5 / +5.6 / +3.9** (normal / surge /
diversion), evidence that the hospital-to-hospital negotiation itself improves outcomes.
Ambulance utilization is slightly lower for the agents (0.84 → 0.82 in normal) while they serve
more people: they avoid spending ambulances and beds on minor injuries while critical victims wait.

### Seed 42: baseline vs. rule agents vs. LLM agents (qwen2.5:3b)

| Metric | Baseline | Rule agents | LLM agents |
|---|---|---|---|
| Victims served (of 50) | 26 | **33** | 22 |
| Critical missed (of 11) | 7 | **6** | 8 |
| Avg response time (min) | 61.4 | 51.2 | **40.4** |
| Critical victim to hospital (min) | 66.4 | 66.7 | 74.9 |
| Ambulance utilization | 0.749 | 0.818 | 0.841 |
| Contested ambulances (conflicts) | – | 10 | 11 |
| Admission requests / accepts | – | 29 / 25 | 23 / 18 |
| Rejects / counter-offers | – | 0 / 4 | 4 / 1 |
| Staff requests / loans | – | 12 / 1 | 3 / 3 |
| Unresolved negotiations | – | 0 | 0 |

### LLM agents: findings

- **Run:** 132 LLM calls in about 91 minutes on a laptop CPU, with 457 tool calls
  (`travel_time` 193, `hospital_options` 175, `incident_status` 54, `check_capacity` 33,
  `triage_keywords` 2).
- **What worked:** the agents used their SDK tools, negotiated admissions and staff loans
  (3 requests, 3 loans), and reached help fastest of all policies (40.4 min response time).
  An earlier LLM run, without tools and pre-computed facts, had its Hospital Agent reject
  44 of 60 admission requests even when beds were free. Giving it the `check_capacity` tool
  and a pre-computed `fits: YES/NO` field cut rejections to 4 and unresolved negotiations
  from 14 to 0.
- **What did not:** 45 of 132 replies (34%) were invalid, mostly because the Dispatch and
  Allocation agents called their tools repeatedly and then ran out of turns or returned no
  JSON. Each invalid reply was caught by Pydantic and replaced by the agent's rule policy, so
  the simulation never broke. Overall the small 3b model served fewer victims than the
  baseline (22 vs 26).
- **Takeaway:** the coordination and negotiation design improves outcomes (rule-mode
  experiments), and the SDK agents implement it end to end. Decision quality with a small
  local model is the limiting factor; a larger model, a cap on tool calls, or guardrails
  around triage priorities should close the gap.

Because a full LLM run takes about 90 minutes locally, the 60-run experiment grid uses the
agents' rule-mode policies; the LLM run is used for comparison.

---

## 6. Dashboard

`python -m streamlit run app.py` opens five tabs, with a run selector, a minute slider and
Play/Pause:

- **Incident map:** schematic city grid with incidents (colour = worst waiting severity,
  size = victims waiting), hospitals with free/total beds, moving ambulances and responder
  teams, and each incident's report.
- **Resources:** ambulances by status, ER beds in use (including surge beds), doctor and nurse
  workload shown separately, staff on duty per hospital, responder teams, and waiting victims,
  free ambulances and free beds over time.
- **Agent activity:** each agent's status, message count, share of LLM decisions and latest
  messages, plus pending admission requests.
- **Negotiation history:** conflicts, requests, accepts, rejects, counter-offers, staff requests
  and staff loans, a chart over time, and the full filterable message log.
- **Performance:** final metrics and negotiation counts for all runs of the seed, plus the
  experiment charts.

---

## 7. Limitations and next steps

- Schematic grid and simplified travel model; no real road network.
- Parameters such as the 60-minute deterioration window and surge-bed counts are assumptions,
  not clinical data.
- Only one responder team of each type, which often becomes the bottleneck.
- Only the "trauma" skill affects staff assignment; other specialties are tracked but not used
  in decisions.
- The small local LLM is slow (about 90 minutes per run) and often loops on tool calls; a larger
  model, a tool-call limit, or triage guardrails should improve the LLM agents.
- One Hospital Coordination Agent speaks for all three hospitals; one agent per hospital would
  make hospital-to-hospital negotiation fully decentralised.