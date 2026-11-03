import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("cache")

WRITER_INSTR = (
    "You write short, realistic emergency dispatch notes (2-3 sentences) as a 911 operator would. "
    "You get the facts as JSON. Describe each injured person's condition in plain language "
    "(e.g. 'trapped and not breathing', 'broken leg', 'minor cuts'). Never use the words "
    "'immediate', 'delayed' or 'minor'. The number of injured people must match the facts exactly. "
    "Return only the note.")


def _facts(incident: dict) -> dict:
    counts = {}
    for v in incident["victims"]:
        counts[v["severity"]] = counts.get(v["severity"], 0) + 1
    return {"type": incident["incident_type"], "victims": len(incident["victims"]),
            "severity_counts": counts,
            "location_km": [incident["location"]["x"], incident["location"]["y"]]}


def _key(facts: dict) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:16]


def _template_report(f: dict) -> str:
    c = f["severity_counts"]
    parts = []
    if c.get("immediate"): parts.append(f"{c['immediate']} unresponsive or with severe bleeding")
    if c.get("delayed"):   parts.append(f"{c['delayed']} with serious but stable injuries")
    if c.get("minor"):     parts.append(f"{c['minor']} walking wounded")
    x, y = f["location_km"]
    return (f"Caller reports a {f['type']} near grid ({x:.1f}, {y:.1f}). "
            f"Approximately {f['victims']} people involved: " + "; ".join(parts) + ".")


class _Writer:

    def __init__(self):
        from openai import AsyncOpenAI
        from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, set_tracing_disabled
        cfg = json.loads(Path("config.json").read_text())["ollama"]
        set_tracing_disabled(True)
        client = AsyncOpenAI(base_url=cfg["base_url"], api_key="ollama", timeout=300)
        model = OpenAIChatCompletionsModel(model=cfg["model"], openai_client=client)
        self.agent = Agent(name="Incident Report Writer", instructions=WRITER_INSTR, model=model,
                           model_settings=ModelSettings(temperature=0))

    def write(self, f: dict) -> str:
        from agents import Runner
        return str(Runner.run_sync(self.agent, json.dumps(f)).final_output).strip()


def add_reports(incidents: list[dict], seed: int, use_llm: bool, model: str = "") -> None:
    if not use_llm:
        for inc in incidents:
            inc["report"] = _template_report(_facts(inc))
        return

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"reports_seed{seed}_llm.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    writer, ok = None, True
    for inc in incidents:
        f = _facts(inc)
        k = _key(f)
        if k not in cache and ok:
            try:
                writer = writer or _Writer()
                print(f"  writing report for {inc['incident_id']}...", flush=True)
                cache[k] = writer.write(f)
            except Exception as e:
                print(f"  ! LLM report failed ({str(e)[:80]}); using the template for the rest")
                ok = False
        inc["report"] = cache.get(k) or _template_report(f)
    cache_file.write_text(json.dumps(cache, indent=2, sort_keys=True))