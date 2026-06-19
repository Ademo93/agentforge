"""AgentForge — ReAct agents on open-weight LLMs with tools and an eval harness."""

from importlib.metadata import PackageNotFoundError, version

from agentforge.core.agent import Agent, AgentResult, Step

try:
    __version__ = version("agentforge-ml")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["Agent", "AgentResult", "Step", "__version__"]
