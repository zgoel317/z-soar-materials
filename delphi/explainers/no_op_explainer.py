import asyncio
from dataclasses import dataclass

from delphi.explainers.explainer import ExplainerResult
from delphi.latents.latents import ActivatingExample, LatentRecord

from .explainer import Explainer


@dataclass
class NoOpExplainer(Explainer):
    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        return ExplainerResult(record=record, explanation="")

    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        return []

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
