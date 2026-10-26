"""OpenAI-compatible gateway. Point any OpenAI client at http://localhost:8000/v1 and set
model to "router-cascade" (default), "router-quality", "router-balanced", "router-0.62", ...

Run: python -m costrouter serve   (or: uvicorn costrouter.serve.app:create_default_app --factory)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from costrouter.serve.service import RouterService


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "router-cascade"
    messages: list[Message]


class RouteRequest(BaseModel):
    text: str
    model: str | None = None


def create_app(service: RouterService) -> FastAPI:
    app = FastAPI(title="Cost-aware LLM router", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "small": service.profile.small.name,
            "large": service.profile.large.name,
            "router": service.router.name,
            "presets": service.presets,
            "cascade": service.cascade,
        }

    @app.post("/route")
    def route(req: RouteRequest) -> dict[str, Any]:
        """The routing decision only, with no LLM call (useful for latency tests)."""
        d = service.decide(req.text, req.model)
        return {
            "p_small": d.p_small,
            "zone": d.zone,
            "would_call": service.profile.large.name
            if d.zone == "large"
            else service.profile.small.name,
            "tau_low": d.tau_low,
            "tau_high": d.tau_high,
            "router_latency_ms": round(d.router_ms, 2),
        }

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> dict[str, Any]:
        if not req.messages:
            raise HTTPException(400, "messages must not be empty")
        out = await service.chat([m.model_dump() for m in req.messages], req.model)
        resp, d = out["response"], out["decision"]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resp.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.text},
                    "finish_reason": resp.stop_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": resp.input_tokens,
                "completion_tokens": resp.output_tokens,
                "total_tokens": resp.input_tokens + resp.output_tokens,
            },
            "x_router": {
                "chosen_model": resp.model,
                "served_by": out["served_by"],
                "p_small": round(d.p_small, 4),
                "zone": d.zone,
                "tau_low": d.tau_low,
                "tau_high": d.tau_high,
                "escalated": out["escalated"],
                "router_latency_ms": round(d.router_ms, 2),
                "total_latency_ms": round(out["latency_ms"], 1),
                "cost_usd": out["cost_usd"],
            },
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return service.metrics()

    return app


def create_default_app() -> FastAPI:
    """Build the app from the trained artifacts of the active (or COSTROUTER_PROFILE) profile."""
    import os

    import torch
    from dotenv import load_dotenv

    from costrouter.config import ROLES, load_models_config, project_root, reports_dir
    from costrouter.llm.factory import build_client

    # The gateway embeds one short query at a time; one thread is faster and steadier for that.
    torch.set_num_threads(1)
    load_dotenv(project_root() / ".env")
    cfg = load_models_config()
    name = os.environ.get("COSTROUTER_PROFILE", cfg.active_profile)
    profile = cfg.profile(name)
    clients = {r: build_client(profile.role(r)) for r in ROLES}
    service = RouterService.from_artifacts(
        project_root() / "models" / name, reports_dir() / name / "results.json", profile, clients
    )
    # Load the embedding model now, not on the first user request (about 30 s cold start).
    service.decide("warm-up")
    return create_app(service)
