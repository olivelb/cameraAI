import gradio as gr
from fastapi.testclient import TestClient
from gradio.routes import App

from streaming import attach_stream


def make_app(auth):
    with gr.Blocks() as blocks:
        gr.Markdown("Stream login test")
    blocks.auth = auth
    app = App.create_app(blocks)
    attach_stream(app, lambda: iter([b"test-frame"]))
    return app


def test_stream_requires_gradio_session_and_rejects_forged_cookie():
    app = make_app([("viewer", "test-password")])
    with TestClient(app) as client:
        assert client.get("/stream").status_code == 401
        client.cookies.set(f"access-token-unsecure-{app.cookie_id}", "forged")
        assert client.get("/stream").status_code == 401
        assert client.post("/login", data={"username": "viewer", "password": "test-password"}).status_code == 200
        reply = client.get("/stream")
        assert reply.status_code == 200
        assert reply.content == b"test-frame"
        assert reply.headers["cache-control"] == "no-store"
        assert client.get("/logout", follow_redirects=False).status_code == 302
        assert client.get("/stream").status_code == 401


def test_stream_available_without_login_in_local_mode():
    with TestClient(make_app(None)) as client:
        assert client.get("/stream").status_code == 200
