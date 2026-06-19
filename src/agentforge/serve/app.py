"""FastAPI app for AgentForge."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentforge.core.agent import Agent


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    max_steps: int | None = Field(None, ge=1, le=20)
    max_new_tokens: int = Field(256, ge=16, le=2048)


class StepModel(BaseModel):
    thought: str
    tool: str | None
    action_input: str | None
    observation: str | None
    elapsed_ms: float


class AskResponse(BaseModel):
    question: str
    final_answer: str
    n_steps: int
    success: bool
    latency_ms: float
    steps: list[StepModel]


class ToolModel(BaseModel):
    name: str
    description: str


def build_app(agent: Agent) -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'FastAPI is required. Install with `pip install "agentforge-ml[serve]"`.'
        ) from e

    app = FastAPI(title="AgentForge", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "n_tools": len(agent.tools)}

    @app.get("/tools", response_model=list[ToolModel])
    def list_tools() -> list[ToolModel]:
        return [ToolModel(name=t.name, description=t.description) for t in agent.tools]

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        result = agent.run(
            req.question,
            max_steps=req.max_steps,
            max_new_tokens=req.max_new_tokens,
        )
        return AskResponse(
            question=result.question,
            final_answer=result.final_answer,
            n_steps=result.n_steps,
            success=result.success,
            latency_ms=result.latency_ms,
            steps=[
                StepModel(
                    thought=s.thought,
                    tool=s.tool,
                    action_input=s.action_input,
                    observation=s.observation,
                    elapsed_ms=s.elapsed_ms,
                )
                for s in result.steps
            ],
        )

    return app
