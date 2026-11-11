"""WebAvatarServer: a local browser preview of the avatar. No Live2D model, no VTube Studio.

    http://127.0.0.1:8765/      the page (web/avatar.html: one self-contained SVG face)
    ws://127.0.0.1:8765/ws      one JSON ExpressionFrame per message, ~30 per second

It exists because VTube Studio needs a rigged model the project does not have yet, and
because it makes the whole chain (emotion -> face, speech -> mouth) visible and testable
today. The browser renders on the integrated GPU, leaving the RTX for the LLM (README
section 4), the same reason VTube Studio is preferred over anything GPU-hungry.

Bound to 127.0.0.1 only: the page needs no network, and nothing here should be reachable
from the LAN.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import WSMsgType, web

from amadeus.expression.frame import ExpressionFrame
from amadeus.trace import get_tracer

PAGE = Path(__file__).with_name("web") / "avatar.html"
# A client that cannot take a frame within this long is dropped, not waited for.
SEND_TIMEOUT_S = 0.25


class WebAvatarServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self._host = host
        self._port = port
        self._tracer = get_tracer("avatar_web")
        self._runner: web.AppRunner | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._latest: str | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ----------------------------------------------------------------- handlers

    async def _index(self, request: web.Request) -> web.Response:
        # Read per request so editing the page needs only a refresh, not a restart.
        return web.Response(text=PAGE.read_text(encoding="utf-8"), content_type="text/html")

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        self._clients.add(ws)
        self._tracer.info(f"Avatar page connected ({len(self._clients)} client(s))")
        try:
            if self._latest is not None:
                await ws.send_str(self._latest)   # draw something immediately, not after the next tick
            async for msg in ws:                  # the page sends nothing; this just detects the close
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    # ---------------------------------------------------------------------- API

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            await web.TCPSite(runner, self._host, self._port).start()
        except OSError as e:
            await runner.cleanup()
            raise RuntimeError(
                f"Could not start the avatar preview on port {self._port}: {e}. "
                "Another program is using it; set avatar.web_port in config/settings.yaml."
            ) from e
        self._runner = runner
        self._tracer.info(f"Avatar preview at {self.url}")

    async def send(self, frame: ExpressionFrame) -> None:
        payload = json.dumps(frame.to_dict(), separators=(",", ":"))
        self._latest = payload
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.send_str(payload), SEND_TIMEOUT_S)
            except Exception:
                self._clients.discard(ws)

    async def close(self) -> None:
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


__all__ = ["WebAvatarServer"]
