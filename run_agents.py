import argparse
import json
import time
from pathlib import Path

from sim.engine import Simulation, load_world
from team.llm import LLM
from team.team import AgentTeam


def main():
    cfg = json.loads(Path("config.json").read_text())
    ollama = cfg.get("ollama", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["rule", "llm"], default="rule")
    ap.add_argument("--model", default=ollama.get("model", "qwen2.5:7b"))
    args = ap.parse_args()

    world = load_world(f"data/world_v1.0_seed{args.seed}.json")
    llm = LLM(enabled=args.mode == "llm", base_url=ollama.get("base_url", "http://localhost:11434/v1"),
              model=args.model)
    team = AgentTeam(llm, mode=args.mode)
    sim = Simulation(world)

    print(f"Running 4 agents ({args.mode} mode{', ' + args.model if args.mode == 'llm' else ''}), seed {args.seed}...")
    t0 = time.time()
    metrics = sim.run(team, f"agents-{args.mode}")
    extra = {"negotiation": team.stats, "messages": team.messages,
             "llm": {"model": args.model if args.mode == "llm" else None, "calls": llm.calls,
                     "failures": llm.failures, "seconds": round(llm.seconds, 1), "last_error": llm.last_error,
                     "tool_calls": llm.tool_calls}}
    out = f"runs/agents_{args.mode}_seed{args.seed}.json"
    sim.save(out, f"agents-{args.mode}", extra)

    base_file = Path(f"runs/baseline_seed{args.seed}.json")
    base = json.loads(base_file.read_text())["metrics"] if base_file.exists() else {}
    print(f"\n  {'metric':30} {'baseline':>10} {'agents':>10}")
    for k, v in metrics.items():
        print(f"  {k:30} {str(base.get(k, '-')):>10} {str(v):>10}")
    print("\nNegotiation:")
    for k, v in team.stats.items():
        print(f"  {k:30} {v}")
    if args.mode == "llm":
        print(f"\nLLM: {llm.calls} calls, {llm.failures} failed (rule fallback used), {llm.seconds:.0f}s")
        print(f"Tool calls: {llm.tool_calls or 'none'}")
        if llm.last_error:
            print(f"  last error: {llm.last_error}")
    print(f"\nSaved {out} ({len(team.messages)} messages) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()