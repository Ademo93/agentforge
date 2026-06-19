# Evaluation

Built-in metrics, pure Python, no judge model required. The four metrics each
isolate one failure mode you want to detect:

| Metric | Failure mode it catches |
|---|---|
| **`task_completion`** | The loop ran out of steps / crashed / went silent |
| **`final_answer_match`** | The agent answered something, but not the right thing |
| **`tool_accuracy`** | The agent answered, but used the wrong tools to get there |
| **`step_efficiency`** | The agent wandered through useless steps |

## Dataset format

JSONL, one object per line:

```jsonl
{"question": "What is 47 * 1337?", "ground_truth": "62839",
 "expected_tools": "calculator", "ground_truth_steps": 2}
```

| Field | Required by |
|---|---|
| `question` | All metrics |
| `ground_truth` | `final_answer_match` |
| `expected_tools` | `tool_accuracy` (string or list) |
| `ground_truth_steps` | `step_efficiency` |

`expected_tools` accepts `"any"` or `"*"` as a wildcard for "any tool is OK
at this step".

## CLI

```bash
af eval examples/eval_set.jsonl --tools all
```

```text
+--------------------+--------+
| metric             | mean   |
+--------------------+--------+
| task_completion    | 0.95   |
| final_answer_match | 0.81   |
| tool_accuracy      | 0.88   |
| step_efficiency    | 0.72   |
+--------------------+--------+
n=80  ·  p50=2.4s  ·  p95=8.1s
```

## Programmatic use

```python
from agentforge import Agent
from agentforge.eval import evaluate
from agentforge.tools import Calculator, PythonREPL

agent = Agent.from_defaults(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    tools=[Calculator(), PythonREPL()],
)

samples = [...]  # list of dicts
results = [agent.run(s["question"]) for s in samples]

res = evaluate(samples, results)
print(res["means"])
```
