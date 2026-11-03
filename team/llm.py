import json
import time


def extract_json(text: str):
    text = text.strip().replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


class LLM:
    def __init__(self, enabled: bool, base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5:7b"):
        self.enabled = enabled
        self.model_name = model
        self.calls = self.failures = 0
        self.seconds = 0.0
        self.last_error = ""
        self.tool_calls = {}       
        self._agents = {}
        if enabled:
            from openai import AsyncOpenAI
            from agents import OpenAIChatCompletionsModel, set_tracing_disabled
            set_tracing_disabled(True)  
            client = AsyncOpenAI(base_url=base_url, api_key="ollama", timeout=300)
            self.model = OpenAIChatCompletionsModel(model=model, openai_client=client)

    def ask(self, name: str, instructions: str, payload: dict, schema, tools=(), sim=None, team=None,
            used=None):
        if not self.enabled:
            return None
        from agents import Agent, Runner, ModelSettings
        from team.tools import ToolContext
        if name not in self._agents:
            self._agents[name] = Agent(name=name, instructions=instructions, model=self.model,
                                       tools=list(tools), model_settings=ModelSettings(temperature=0))
        context = ToolContext(sim=sim, team=team, used=dict(used or {}), calls=self.tool_calls)
        self.calls += 1
        print(f"  [{self.calls}] asking {name}...", flush=True)
        t0 = time.time()
        try:
            result = Runner.run_sync(self._agents[name], json.dumps(payload), context=context, max_turns=6)
            return schema.model_validate(extract_json(str(result.final_output)))
        except Exception as e:                      
            self.failures += 1
            self.last_error = f"{name}: {str(e)[:200]}"
            return None
        finally:
            self.seconds += time.time() - t0