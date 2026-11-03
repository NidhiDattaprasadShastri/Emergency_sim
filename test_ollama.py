import json
from pathlib import Path

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

cfg = json.loads(Path("config.json").read_text())["ollama"]
set_tracing_disabled(True)
client = AsyncOpenAI(base_url=cfg["base_url"], api_key="ollama")
model = OpenAIChatCompletionsModel(model=cfg["model"], openai_client=client)

agent = Agent(name="Test Agent", model=model,
              instructions="You coordinate emergency response resources in a simulation. Answer in 2 sentences.")
result = Runner.run_sync(agent, "Why do ambulance availability and hospital capacity matter in an emergency?")
print(f"Model {cfg['model']} replied:\n{result.final_output}")