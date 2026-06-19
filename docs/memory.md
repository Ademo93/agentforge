# Memory

Two backends, same three-method interface.

## `ConversationMemory`

In-process FIFO window. Good for chat-style sessions inside a single Python
process.

```python
from agentforge.memory import ConversationMemory

mem = ConversationMemory(max_messages=64)
mem.add("user", "Hello")
mem.add("assistant", "Hi there!")
for msg in mem.get(limit=10):
    print(msg.role, msg.content)
```

## `PersistentMemory`

SQLite-backed, session-keyed. Survives process restarts and is multi-user
safe (each `session` id has its own history).

```python
from agentforge.memory import PersistentMemory

mem = PersistentMemory("agentforge_memory.db", session="alice")
mem.add("user", "Remember I prefer terse answers", source="cli")
mem.add("assistant", "Got it.")

# Later, possibly in a different process:
hist = PersistentMemory("agentforge_memory.db", session="alice").get()
```

The schema is one table, `messages(id, session, role, content, ts, extras_json)`,
with an index on `(session, ts)`. That is enough for ~10⁶ messages per
session before query latency matters.

## Plug in your own

Implement the `Memory` protocol — `add`, `get`, `clear`. Useful targets:

- Redis for ephemeral multi-process sessions
- Postgres + `pgvector` if you want semantic recall over the history
- A wrapper around `ragforge.Pipeline` so the agent can search its own past
