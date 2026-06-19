"""FastAPI smoke test."""

from __future__ import annotations

import pytest

from agentforge.core.agent import Agent
from agentforge.tools import Calculator


@pytest.fixture
def client(scripted_llm):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from agentforge.serve import build_app

    llm = scripted_llm(
        [
            "Thought: trivial\nFinal Answer: 4",
        ]
    )
    agent = Agent(llm=llm, tools=[Calculator()], max_steps=3)
    return TestClient(build_app(agent))


def test_health(client) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["n_tools"] == 1


def test_tools_endpoint(client) -> None:
    res = client.get("/tools")
    assert res.status_code == 200
    body = res.json()
    assert any(t["name"] == "calculator" for t in body)


def test_ask_endpoint(client) -> None:
    res = client.post("/ask", json={"question": "2 + 2 ?"})
    assert res.status_code == 200
    body = res.json()
    assert body["final_answer"] == "4"
    assert body["success"] is True
