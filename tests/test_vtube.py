"""VTubeStudioSink against a MOCK VTube Studio that speaks the documented protocol.

This proves the message flow and the value maths. It cannot prove that a particular Live2D
model reacts well: the project has no model, and no real VTube Studio was available.
"""

import asyncio
import json
import socket

import pytest
from aiohttp import WSMsgType, web

from amadeus.expression.frame import ExpressionFrame
from amadeus.expression.vtube import CUSTOM_PARAMS, MAPPINGS, ParamRange, VTubeStudioSink, scale

DEFAULT_PARAMS = {
    # name: (min, max, default)
    "MouthOpen": (0, 1, 0), "MouthSmile": (0, 1, 0.5), "EyeOpenLeft": (0, 1, 1), "EyeOpenRight": (0, 1, 1),
    "Brows": (0, 1, 0.5), "FaceAngleX": (-30, 30, 0), "FaceAngleY": (-30, 30, 0), "FaceAngleZ": (-30, 30, 0),
    "EyeLeftX": (-1, 1, 0), "EyeLeftY": (-1, 1, 0), "EyeRightX": (-1, 1, 0), "EyeRightY": (-1, 1, 0),
}


class MockVTS:
    def __init__(self, token="tok-abc", allow=True, params=None):
        self.token, self.allow = token, allow
        self.params = dict(DEFAULT_PARAMS if params is None else params)
        self.custom = {}
        self.received = []
        self.runner = None
        self.port = None
        self.sockets = []

    def of(self, message_type):
        return [m for m in self.received if m["messageType"] == message_type]

    async def _handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.append(ws)
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            self.received.append(m)
            t, d = m["messageType"], m.get("data", {})
            reply = lambda mt, data: ws.send_json({"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
                                                   "requestID": m["requestID"], "messageType": mt, "data": data})
            if t == "AuthenticationTokenRequest":
                if self.allow:
                    await reply("AuthenticationTokenResponse", {"authenticationToken": self.token})
                else:
                    await reply("APIError", {"errorID": 50, "message": "Token request denied"})
            elif t == "AuthenticationRequest":
                await reply("AuthenticationResponse", {"authenticated": d.get("authenticationToken") == self.token,
                                                       "reason": ""})
            elif t == "InputParameterListRequest":
                row = lambda n, v: {"name": n, "addedBy": "x", "value": 0, "min": v[0], "max": v[1], "defaultValue": v[2]}
                await reply("InputParameterListResponse", {
                    "modelLoaded": True, "modelName": "Test",
                    "defaultParameters": [row(n, v) for n, v in self.params.items()],
                    "customParameters": [row(n, v) for n, v in self.custom.items()]})
            elif t == "ParameterCreationRequest":
                self.custom[d["parameterName"]] = (d["min"], d["max"], d["defaultValue"])
                await reply("ParameterCreationResponse", {"parameterName": d["parameterName"]})
            # Inject and hotkey requests are fire-and-forget in the sink; nothing to answer.
        return ws

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self._handler)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port or 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def stop(self):
        # Like the real app quitting: drop live connections, do not wait for them to finish.
        for ws in self.sockets:
            await ws.close()
        self.sockets.clear()
        if self.runner:
            await self.runner.cleanup()
            self.runner = None

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"


async def wait_for(cond, timeout=3.0):
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


def run(coro, timeout=15):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def make_sink(mock, tmp_path, **kw):
    return VTubeStudioSink(url=mock.url, token_path=tmp_path / "token.txt", reconnect_s=0.05, **kw)


# ------------------------------------------------------------------------ maths

def test_unit_scaling_maps_zero_one_onto_min_max():
    r = ParamRange(2.0, 10.0, 2.0)
    assert scale("unit", 0.0, r) == 2.0 and scale("unit", 1.0, r) == 10.0 and scale("unit", 0.5, r) == 6.0
    assert scale("unit", 9.0, r) == 10.0 and scale("unit", -3.0, r) == 2.0     # clamped
    assert scale("unit", 0.25, r, invert=True) == pytest.approx(8.0)


def test_signed_scaling_is_centred_on_the_parameters_own_default():
    smile = ParamRange(0.0, 1.0, 0.5)               # neutral is the MIDDLE, not the minimum
    assert scale("signed", 0.0, smile) == 0.5
    assert scale("signed", 1.0, smile) == 1.0 and scale("signed", -1.0, smile) == 0.0
    angle = ParamRange(-30.0, 30.0, 0.0)
    assert scale("signed", 0.5, angle) == 15.0 and scale("signed", -0.5, angle) == -15.0
    skewed = ParamRange(0.0, 10.0, 2.0)             # neutral is NOT the midpoint (5.0): it is 2.0
    assert scale("signed", 0.0, skewed) == 2.0
    assert scale("signed", 1.0, skewed) == 10.0 and scale("signed", -1.0, skewed) == 0.0
    assert scale("signed", 0.5, skewed) == 6.0 and scale("signed", -0.5, skewed) == 1.0   # each side uses its own room
    assert scale("signed", 0.5, angle, invert=True) == -15.0
    assert scale("signed", 5.0, angle) == 30.0      # clamped


# ----------------------------------------------------------------- auth / setup

def test_first_run_asks_for_a_token_saves_it_and_starts_driving(tmp_path):
    async def go():
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            assert (tmp_path / "token.txt").read_text() == "tok-abc"
            assert len(mock.of("AuthenticationTokenRequest")) == 1
            req = mock.of("AuthenticationTokenRequest")[0]
            assert req["apiName"] == "VTubeStudioPublicAPI" and req["apiVersion"] == "1.0" and req["requestID"]
            assert req["data"]["pluginName"] == "Amadeus"
            assert mock.of("AuthenticationRequest")[0]["data"]["authenticationToken"] == "tok-abc"
            # the custom parameters were created for the user to map
            assert set(mock.custom) == set(CUSTOM_PARAMS)
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_a_saved_token_is_reused_without_another_popup(tmp_path):
    async def go():
        (tmp_path / "token.txt").write_text("tok-abc")
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            assert mock.of("AuthenticationTokenRequest") == []      # the user was NOT asked again
            assert len(mock.of("AuthenticationRequest")) == 1
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_a_revoked_token_is_replaced(tmp_path):
    async def go():
        (tmp_path / "token.txt").write_text("stale-token")
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            assert len(mock.of("AuthenticationTokenRequest")) == 1
            assert (tmp_path / "token.txt").read_text() == "tok-abc"
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_denying_access_leaves_the_sink_quiet_and_harmless(tmp_path):
    async def go():
        mock = MockVTS(allow=False); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        try:
            await wait_for(lambda: mock.of("AuthenticationTokenRequest"))
            await asyncio.sleep(0.2)
            await sink.send(ExpressionFrame(mouth_open=1.0))         # must not raise
            assert not sink.ready and mock.of("InjectParameterDataRequest") == []
            assert not (tmp_path / "token.txt").exists()
            assert len(mock.of("AuthenticationTokenRequest")) == 1   # and does not nag every 50 ms
        finally:
            await sink.close(); await mock.stop()
    run(go())


# ---------------------------------------------------------------------- driving

def test_frames_are_injected_scaled_into_each_parameters_own_range(tmp_path):
    async def go():
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            await sink.send(ExpressionFrame(mouth_open=0.5, mouth_smile=1.0, eye_open=0.8, head_yaw=15.0,
                                            gaze_x=-0.5, blush=0.4, brow_inner=-1.0))
            await wait_for(lambda: mock.of("InjectParameterDataRequest"))
            msg = mock.of("InjectParameterDataRequest")[-1]
            assert msg["data"]["mode"] == "set" and msg["data"]["faceFound"] is True
            v = {p["id"]: p["value"] for p in msg["data"]["parameterValues"]}
            assert v["MouthOpen"] == 0.5
            assert v["MouthSmile"] == 1.0                            # smile +1 -> top of its range
            assert v["EyeOpenLeft"] == v["EyeOpenRight"] == 0.8
            assert v["FaceAngleX"] == 15.0                           # 15 deg of +-30
            assert v["EyeLeftX"] == -0.5
            assert v["AmadeusBlush"] == 0.4 and v["AmadeusBrowInner"] == -1.0
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_parameters_the_model_does_not_have_are_skipped_not_invented(tmp_path):
    async def go():
        params = {k: v for k, v in DEFAULT_PARAMS.items() if k not in ("Brows", "FaceAngleZ")}
        mock = MockVTS(params=params); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            assert "Brows" not in sink.driven_parameters and "FaceAngleZ" not in sink.driven_parameters
            await sink.send(ExpressionFrame())
            await wait_for(lambda: mock.of("InjectParameterDataRequest"))
            ids = {p["id"] for p in mock.of("InjectParameterDataRequest")[-1]["data"]["parameterValues"]}
            assert "Brows" not in ids and "MouthOpen" in ids
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_invert_flips_a_parameter_that_points_the_wrong_way(tmp_path):
    async def go():
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path, invert=["FaceAngleY"]); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            vals = {p["id"]: p["value"] for p in sink.parameter_values(ExpressionFrame(head_pitch=15.0))}
            assert vals["FaceAngleY"] == -15.0 and vals["FaceAngleX"] == 0.0
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_a_hotkey_fires_once_when_the_mood_changes(tmp_path):
    async def go():
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path, hotkeys={"flustered": "Blush", "neutral": "Reset"}); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            await sink.send(ExpressionFrame(label="neutral"))
            for _ in range(5):                                       # same mood, five frames
                await sink.send(ExpressionFrame(label="flustered"))
            await sink.send(ExpressionFrame(label="annoyed"))        # no hotkey configured for this
            await wait_for(lambda: len(mock.of("HotkeyTriggerRequest")) >= 2)
            await asyncio.sleep(0.1)
            assert [m["data"]["hotkeyID"] for m in mock.of("HotkeyTriggerRequest")] == ["Reset", "Blush"]
        finally:
            await sink.close(); await mock.stop()
    run(go())


# ---------------------------------------------------------------- availability

def test_no_vtube_studio_running_costs_nothing_and_connects_when_it_appears(tmp_path):
    async def go():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        mock = MockVTS(); mock.port = port                           # not started yet
        sink = make_sink(mock, tmp_path)
        await sink.start()                                           # must return immediately
        for _ in range(20):
            await sink.send(ExpressionFrame())                       # silently dropped
        assert not sink.ready
        await mock.start()                                           # VTube Studio is launched later
        try:
            await wait_for(lambda: sink.ready, timeout=5)
            await sink.send(ExpressionFrame(mouth_open=1.0))
            await wait_for(lambda: mock.of("InjectParameterDataRequest"))
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_it_reconnects_after_vtube_studio_restarts(tmp_path):
    async def go():
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        try:
            await wait_for(lambda: sink.ready)
            await mock.stop()                                        # VTube Studio quits
            await wait_for(lambda: not sink.ready, timeout=5)
            mock.received.clear()
            await mock.start()                                       # and is relaunched on the same port
            await wait_for(lambda: sink.ready, timeout=8)
            assert mock.of("AuthenticationTokenRequest") == []       # the saved token still works
        finally:
            await sink.close(); await mock.stop()
    run(go())


def test_close_is_prompt_even_while_still_connecting(tmp_path):
    async def go():
        mock = MockVTS(); await mock.start()
        sink = make_sink(mock, tmp_path); await sink.start()
        await asyncio.wait_for(sink.close(), 3.0)
        await mock.stop()
    run(go())


def test_every_mapped_parameter_is_either_a_known_default_or_a_declared_custom_one():
    """Guards against a typo in MAPPINGS silently driving nothing."""
    names = {n for n, _, _ in MAPPINGS}
    assert names - set(DEFAULT_PARAMS) == set(CUSTOM_PARAMS)
