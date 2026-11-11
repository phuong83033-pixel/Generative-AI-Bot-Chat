"""WebAvatarServer: the local browser preview. Real aiohttp server, real WebSocket clients."""

import asyncio
import json
import socket

import aiohttp
import pytest

from amadeus.expression.frame import ExpressionFrame
from amadeus.expression.server import PAGE, WebAvatarServer


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(coro, timeout=10):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def test_page_is_served_and_is_the_real_avatar():
    async def go():
        srv = WebAvatarServer(port=free_port())
        await srv.start()
        try:
            async with aiohttp.ClientSession() as s:
                r = await s.get(srv.url)
                return r.status, r.content_type, await r.text()
        finally:
            await srv.close()

    status, ctype, body = run(go())
    assert status == 200 and ctype == "text/html"
    assert "<svg" in body and 'id="mouth"' in body and "/ws" in body
    assert body == PAGE.read_text(encoding="utf-8")


def test_frames_reach_every_connected_client_as_json():
    async def go():
        srv = WebAvatarServer(port=free_port())
        await srv.start()
        try:
            async with aiohttp.ClientSession() as s:
                a = await s.ws_connect(srv.url + "ws")
                b = await s.ws_connect(srv.url + "ws")
                await asyncio.sleep(0.05)
                await srv.send(ExpressionFrame(mouth_open=0.7, label="flustered", state="SPEAKING"))
                ma, mb = await asyncio.gather(a.receive_json(timeout=2), b.receive_json(timeout=2))
                return srv.client_count, ma, mb
        finally:
            await srv.close()

    n, ma, mb = run(go())
    assert n == 2
    assert ma == mb and ma["mouth_open"] == 0.7 and ma["label"] == "flustered" and ma["state"] == "SPEAKING"


def test_a_page_opened_late_gets_the_current_face_immediately():
    async def go():
        srv = WebAvatarServer(port=free_port())
        await srv.start()
        try:
            await srv.send(ExpressionFrame(label="annoyed"))        # nobody was listening
            async with aiohttp.ClientSession() as s:
                ws = await s.ws_connect(srv.url + "ws")
                return await ws.receive_json(timeout=2)             # no further send(): must still arrive
        finally:
            await srv.close()

    assert run(go())["label"] == "annoyed"


def test_a_disconnected_client_is_forgotten_and_does_not_break_sending():
    async def go():
        srv = WebAvatarServer(port=free_port())
        await srv.start()
        try:
            async with aiohttp.ClientSession() as s:
                ws = await s.ws_connect(srv.url + "ws")
                await asyncio.sleep(0.05)
                await ws.close()
                await asyncio.sleep(0.1)
                await srv.send(ExpressionFrame())                  # must not raise
                return srv.client_count
        finally:
            await srv.close()

    assert run(go()) == 0


def test_sending_with_no_clients_is_a_cheap_noop():
    async def go():
        srv = WebAvatarServer(port=free_port())
        await srv.start()
        try:
            for _ in range(100):
                await srv.send(ExpressionFrame())
            return srv.client_count
        finally:
            await srv.close()

    assert run(go()) == 0


def test_port_in_use_gives_an_actionable_error():
    async def go():
        port = free_port()
        a = WebAvatarServer(port=port)
        await a.start()
        try:
            with pytest.raises(RuntimeError, match="avatar.web_port"):
                await WebAvatarServer(port=port).start()
        finally:
            await a.close()

    run(go())


def test_only_the_loopback_interface_is_bound():
    """The preview must not be reachable from the LAN."""
    srv = WebAvatarServer(port=1234)
    assert srv.url.startswith("http://127.0.0.1:")
