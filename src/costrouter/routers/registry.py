from __future__ import annotations

from costrouter.routers.base import Router
from costrouter.routers.baselines import LengthRouter, RandomRouter
from costrouter.routers.sklearn_routers import default_routers


def all_routers(seed: int = 42) -> list[Router]:
    return [RandomRouter(seed), LengthRouter(), *default_routers(seed)]
