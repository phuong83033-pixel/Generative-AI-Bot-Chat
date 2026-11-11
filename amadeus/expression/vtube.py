"""VTubeStudioSink: drive a Live2D model in VTube Studio from ExpressionFrames.

Protocol (checked against the official README, github.com/DenchiSoft/VTubeStudio):
  * WebSocket, default ws://localhost:8001. Every message is JSON:
      {"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0", "requestID": ..., "messageType": ..., "data": {...}}
  * First run: AuthenticationTokenRequest -> the USER must click "Allow" in VTube Studio ->
    a token comes back. It is saved and reused: AuthenticationRequest on later runs.
  * InputParameterListRequest returns every tracking parameter with its min / max / default.
  * InjectParameterDataRequest sets parameters. Each must be resent at least once a second
    or VTube Studio hands it back, which is why this sink sends every frame (30 Hz).
  * Custom parameters (name: 4-32 alphanumerics) are made with ParameterCreationRequest.

NOT verified against a real VTube Studio: the project has no Live2D model yet. This file is
tested against a mock server that speaks the documented protocol (tests/test_vtube.py), which
proves the message flow and the maths but not that a particular model reacts well.

Two choices to avoid guessing:
  * The default parameter names I use (MouthOpen, MouthSmile, EyeOpenLeft, FaceAngleX, ...) are
    the ones I know VTube Studio ships, but the README does not list them. So the sink asks
    VTube Studio which parameters exist and only drives those; names it does not have are logged.
  * Value ranges are not hard-coded. Each frame value is scaled into the parameter's own
    reported min / max / default.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import aiohttp

from amadeus.expression.frame import ExpressionFrame
from amadeus.trace import get_tracer

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"
_AUTH_DENIED_ERROR_ID = 50  # documented: the user clicked "Deny" (or ignored the popup)

CUSTOM_PARAMS = {
    "AmadeusBlush": ("Embarrassment, 0 to 1. Map it to a blush layer or a face-flush expression.", 0.0, 1.0, 0.0),
    "AmadeusBrowInner": ("-1 angry (inner brows down) to +1 worried (inner brows up).", -1.0, 1.0, 0.0),
}

# (VTube Studio parameter, kind, value from the frame).
#   unit   : x in 0..1   -> min + x * (max - min)
#   signed : x in -1..1  -> default + x * (max - default)   for x >= 0
#                           default + x * (default - min)   for x <  0
# Head angles are in degrees; +-30 degrees is full deflection.
MAPPINGS: list[tuple[str, str, Callable[[ExpressionFrame], float]]] = [
    ("MouthOpen", "unit", lambda f: f.mouth_open),
    ("MouthSmile", "signed", lambda f: f.mouth_smile),
    ("EyeOpenLeft", "unit", lambda f: f.eye_open),
    ("EyeOpenRight", "unit", lambda f: f.eye_open),
    ("Brows", "signed", lambda f: f.brow_raise),
    ("FaceAngleX", "signed", lambda f: f.head_yaw / 30.0),
    ("FaceAngleY", "signed", lambda f: f.head_pitch / 30.0),
    ("FaceAngleZ", "signed", lambda f: f.head_roll / 30.0),
    ("EyeLeftX", "signed", lambda f: f.gaze_x),
    ("EyeLeftY", "signed", lambda f: f.gaze_y),
    ("EyeRightX", "signed", lambda f: f.gaze_x),
    ("EyeRightY", "signed", lambda f: f.gaze_y),
    ("AmadeusBlush", "unit", lambda f: f.blush),
    ("AmadeusBrowInner", "signed", lambda f: f.brow_inner),
]


@dataclass(frozen=True)
class ParamRange:
    min: float
    max: float
    default: float


def scale(kind: str, x: float, r: ParamRange, invert: bool = False) -> float:
    """Map a renderer-independent value into one parameter's own range."""
    if kind == "unit":
        x = min(1.0, max(0.0, x))
        if invert:
            x = 1.0 - x
        return r.min + x * (r.max - r.min)
    x = min(1.0, max(-1.0, x))
    if invert:
        x = -x
    return r.default + x * ((r.max - r.default) if x >= 0 else (r.default - r.min))


class VTubeAuthDenied(RuntimeError):
    pass


class VTubeStudioSink:
    def __init__(
        self,
        url: str = "ws://127.0.0.1:8001",
        plugin_name: str = "Amadeus",
        plugin_developer: str = "Amadeus project",
        token_path: str | Path = "data/vtube_token.txt",
        hotkeys: dict[str, str] | None = None,
        invert: list[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        reconnect_s: float = 3.0,
    ):
        self._url = url
        self._plugin = plugin_name
        self._developer = plugin_developer
        self._token_path = Path(token_path)
        self._hotkeys = {k.lower(): v for k, v in (hotkeys or {}).items()}
        self._invert = set(invert or [])
        self._clock = clock
        self._reconnect_s = reconnect_s
        self._tracer = get_tracer("avatar_vtube")

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._ranges: dict[str, ParamRange] = {}
        self._ready = False
        self._closing = False
        self._last_label: str | None = None
        self._last_warn: dict[str, float] = {}

    # ------------------------------------------------------------------ status

    @property
    def ready(self) -> bool:
        """Authenticated and parameters resolved: frames are being delivered."""
        return self._ready

    @property
    def driven_parameters(self) -> list[str]:
        return [n for n, _, _ in MAPPINGS if n in self._ranges]

    def _warn_once_per(self, key: str, msg: str, every: float = 30.0) -> None:
        now = self._clock()
        if now - self._last_warn.get(key, -1e9) >= every:
            self._last_warn[key] = now
            self._tracer.warning(msg)

    # ----------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Begin connecting in the background. Returns immediately: VTube Studio may not be
        running, and the avatar loop must not wait for it."""
        self._task = asyncio.create_task(self._connector(), name="vtube-connector")

    async def close(self) -> None:
        self._closing = True
        self._ready = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._drop_connection()

    async def _drop_connection(self) -> None:
        self._ready = False
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
        self._reader_task = None
        self._pending.clear()
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ---------------------------------------------------------------- connection

    async def _connector(self) -> None:
        while not self._closing:
            wait = self._reconnect_s
            try:
                await self._connect_and_authenticate()
                self._tracer.info(
                    f"VTube Studio ready: driving {len(self.driven_parameters)} parameters "
                    f"({', '.join(self.driven_parameters)})"
                )
                reader = self._reader_task
                await reader                      # returns when the socket closes
                self._warn_once_per("lost", "VTube Studio connection closed; will reconnect")
            except asyncio.CancelledError:
                raise
            except VTubeAuthDenied as e:
                self._warn_once_per("denied", str(e), every=60.0)
                wait = 60.0                       # do not nag with a popup every 3 seconds
            except Exception as e:
                self._warn_once_per("connect", f"VTube Studio unavailable ({type(e).__name__}: {e}); retrying")
            await self._drop_connection()
            await asyncio.sleep(wait)

    async def _connect_and_authenticate(self) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(sock_connect=3.0))
        self._ws = await self._session.ws_connect(self._url, heartbeat=20)
        self._reader_task = asyncio.create_task(self._reader(self._ws))

        token = self._load_token()
        authed = False
        if token:
            authed = await self._authenticate(token)
            if not authed:
                self._tracer.info("Saved VTube Studio token was rejected; asking for a new one")
                self._token_path.unlink(missing_ok=True)
        if not authed:
            self._tracer.info(
                "Asking VTube Studio for access: click ALLOW in the popup inside VTube Studio (60 s)."
            )
            token = await self._request_token()
            self._save_token(token)
            if not await self._authenticate(token):
                raise RuntimeError("VTube Studio rejected the token it just issued")

        reply = await self._request("InputParameterListRequest")
        data = reply.get("data", {})
        listed = {p["name"]: p for p in (data.get("defaultParameters") or []) + (data.get("customParameters") or [])}
        for name, (explanation, lo, hi, default) in CUSTOM_PARAMS.items():
            if name not in listed:
                await self._request("ParameterCreationRequest", {
                    "parameterName": name, "explanation": explanation, "min": lo, "max": hi, "defaultValue": default,
                })
                listed[name] = {"name": name, "min": lo, "max": hi, "defaultValue": default}
        self._ranges = {
            n: ParamRange(float(p["min"]), float(p["max"]), float(p.get("defaultValue", 0.0)))
            for n, p in listed.items() if n in {m[0] for m in MAPPINGS}
        }
        missing = [n for n, _, _ in MAPPINGS if n not in self._ranges]
        if missing:
            self._tracer.info(f"This model/VTube Studio has no parameter(s) {', '.join(missing)}: skipped")
        self._ready = True

    async def _authenticate(self, token: str) -> bool:
        reply = await self._request("AuthenticationRequest", {
            "pluginName": self._plugin, "pluginDeveloper": self._developer, "authenticationToken": token,
        })
        return bool(reply.get("data", {}).get("authenticated"))

    async def _request_token(self) -> str:
        reply = await self._request("AuthenticationTokenRequest", {
            "pluginName": self._plugin, "pluginDeveloper": self._developer,
        }, timeout=60.0)
        token = reply.get("data", {}).get("authenticationToken")
        if not token:
            raise RuntimeError(f"No token in reply: {reply}")
        return token

    def _load_token(self) -> str | None:
        try:
            text = self._token_path.read_text(encoding="utf-8").strip()
            return text or None
        except FileNotFoundError:
            return None

    def _save_token(self, token: str) -> None:
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(token, encoding="utf-8")

    # ------------------------------------------------------------------ messages

    def _envelope(self, message_type: str, data: dict | None, request_id: str) -> str:
        msg = {"apiName": API_NAME, "apiVersion": API_VERSION, "requestID": request_id, "messageType": message_type}
        if data is not None:
            msg["data"] = data
        return json.dumps(msg, separators=(",", ":"))

    async def _request(self, message_type: str, data: dict | None = None, timeout: float = 5.0) -> dict:
        """Send a request and wait for the reply with the same requestID."""
        rid = uuid.uuid4().hex[:16]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._ws.send_str(self._envelope(message_type, data, rid))
            reply = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)
        if reply.get("messageType") == "APIError":
            err = reply.get("data", {})
            if err.get("errorID") == _AUTH_DENIED_ERROR_ID:
                raise VTubeAuthDenied(
                    "VTube Studio access was denied. Click ALLOW in the plugin popup inside VTube Studio."
                )
            raise RuntimeError(f"{message_type} failed: {err.get('message', err)}")
        return reply

    async def _reader(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await self._read_loop(ws)
        finally:
            # The socket is gone: fail anything still waiting now, not after its timeout.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(ConnectionError("VTube Studio closed the connection"))

    async def _read_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
                continue
            try:
                reply = json.loads(msg.data)
            except ValueError:
                continue
            fut = self._pending.get(reply.get("requestID"))
            if fut is not None and not fut.done():
                fut.set_result(reply)
            elif reply.get("messageType") == "APIError":     # an error for a fire-and-forget inject
                self._warn_once_per("inject", f"VTube Studio reported: {reply.get('data', {}).get('message')}")

    # ---------------------------------------------------------------------- sink

    def parameter_values(self, frame: ExpressionFrame) -> list[dict]:
        """The InjectParameterDataRequest payload for this frame (pure; used by tests)."""
        out = []
        for name, kind, get in MAPPINGS:
            r = self._ranges.get(name)
            if r is not None:
                out.append({"id": name, "value": round(scale(kind, get(frame), r, name in self._invert), 4)})
        return out

    async def send(self, frame: ExpressionFrame) -> None:
        if not self._ready or self._ws is None or self._ws.closed:
            return
        values = self.parameter_values(frame)
        if values:
            await asyncio.wait_for(self._ws.send_str(self._envelope(
                "InjectParameterDataRequest",
                {"faceFound": True, "mode": "set", "parameterValues": values},
                uuid.uuid4().hex[:16],
            )), 0.25)
        if frame.label != self._last_label:
            self._last_label = frame.label
            hotkey = self._hotkeys.get(frame.label.lower())
            if hotkey:
                await self._ws.send_str(self._envelope("HotkeyTriggerRequest", {"hotkeyID": hotkey}, uuid.uuid4().hex[:16]))


__all__ = ["VTubeStudioSink", "scale", "ParamRange", "MAPPINGS", "VTubeAuthDenied"]
