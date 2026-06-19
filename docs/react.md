# The ReAct loop

```text
question -> [LLM] Thought + Action -> [Tool] Observation
            ^                                       |
            |_______________________________________|
                       up to max_steps
```

ReAct alternates **Reasoning** (Thought + Action) with **Action**
(observation from a tool) until the model emits a `Final Answer:`. It is the
simplest agent pattern that works on small open models — and the easiest to
debug.

## The loop, in code

```python
for step in range(max_steps):
    prompt = system + question + scratchpad_of_prior_steps
    raw = llm.generate(prompt, stop=["\nObservation:"])
    parsed = parse_step(raw)

    if parsed.is_final:
        return AgentResult(final=parsed.final_answer, steps=...)

    tool = registry.get(parsed.tool)
    observation = tool.run(parsed.action_input)
    scratchpad.append((parsed, observation))
```

That is it. Everything in [`agent.py`](https://github.com/Ademo93/agentforge/blob/main/src/agentforge/core/agent.py) is plumbing around this.

## Prompt format

```text
Thought: <reasoning>
Action: <tool name>
Action Input: <input string>
Observation: <tool output, supplied by the loop>
... (repeat) ...
Thought: <final reasoning>
Final Answer: <concise answer>
```

The parser is forgiving: it tolerates case, extra whitespace, and a single
extra line after `Final Answer:`. If the LLM fails to emit an Action at all,
the loop treats the raw response as the answer (graceful degradation).

## Stopping

Two ways the loop terminates:

1. **`Final Answer:`** — `result.success = True`
2. **`max_steps` reached** — `result.success = False`, the final answer
   falls back to the last observation.

## Tuning

| Symptom | First thing to try |
|---|---|
| Loop never terminates | Reduce `max_steps`, or use a stronger model |
| Tool not called | Inspect the system prompt — does the description match how the LLM thinks about the task? |
| Tool called with wrong args | Add a one-line example in the tool's `description` |
| Answer is verbose | Add a `"answer in one sentence"` instruction in your question |
| LLM emits two actions at once | The parser only takes the first — make the prompt mention this explicitly |
