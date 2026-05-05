"""HTTP + CLI client for interacting with KrillLM."""

from __future__ import annotations

import asyncio
import shutil
import json
from collections.abc import AsyncGenerator, AsyncIterator

import httpx

from config import KRILLM_BASE_URL, KRILLM_BIN


class KrillLMClient:
    """Async wrapper around the KrillLM HTTP API and CLI."""

    def __init__(self, base_url: str = KRILLM_BASE_URL) -> None:
        self.base_url = base_url
        self._http = httpx.AsyncClient(base_url=base_url, timeout=300)

    # ── Health / Status ──────────────────────────────────────────────

    async def health(self) -> dict:
        r = await self._http.get("/healthz")
        r.raise_for_status()
        return r.json()

    async def status(self) -> dict:
        r = await self._http.get("/v1/status")
        r.raise_for_status()
        return r.json()

    # ── Model Management ─────────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        r = await self._http.get("/v1/models")
        r.raise_for_status()
        return r.json().get("data", [])

    async def load_model(self, model: str) -> dict:
        r = await self._http.post("/v1/models/load", json={"model": model})
        r.raise_for_status()
        return r.json()

    async def unload_model(self) -> dict:
        r = await self._http.post("/v1/models/unload")
        r.raise_for_status()
        return r.json()

    # ── Inference ─────────────────────────────────────────────────────

    async def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = await self._http.post(
            "/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with self._http.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    return
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if text := delta.get("content"):
                    yield text

    # ── CLI operations (pull, rm, bench) ─────────────────────────────

    @staticmethod
    def _find_cli() -> str:
        path = shutil.which(KRILLM_BIN)
        if not path:
            raise FileNotFoundError(
                f"krillm CLI not found. Set KRILLM_BIN or install to PATH."
            )
        return path

    async def pull_model(self, model: str) -> AsyncGenerator[str, None]:
        """Pull a model via CLI, yielding stdout lines as progress."""
        cli = self._find_cli()
        proc = await asyncio.create_subprocess_exec(
            cli, "pull", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            yield raw.decode().rstrip()
        await proc.wait()

    async def remove_model(self, model: str) -> str:
        cli = self._find_cli()
        proc = await asyncio.create_subprocess_exec(
            cli, "rm", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return out.decode().strip()

    async def bench_model(self, model: str) -> str:
        cli = self._find_cli()
        proc = await asyncio.create_subprocess_exec(
            cli, "bench", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return out.decode().strip()

    async def close(self) -> None:
        await self._http.aclose()
