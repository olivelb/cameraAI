"""Attach the live stream using the same login dependency as Gradio."""

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse


def attach_stream(app, frames) -> None:
    # Use the same session store and cookies as Gradio's /login_check.
    # Covered by real login/logout HTTP tests against the pinned Gradio version.
    if not all(hasattr(app, attr) for attr in ("auth", "auth_dependency", "tokens", "cookie_id")):
        raise RuntimeError("Unsupported Gradio app: session store unavailable")

    def login_check(request: Request):
        if app.auth is None and app.auth_dependency is None:
            return
        if app.auth_dependency is not None:
            user = app.auth_dependency(request)
        else:
            token = request.cookies.get(f"access-token-{app.cookie_id}") or request.cookies.get(
                f"access-token-unsecure-{app.cookie_id}"
            )
            user = app.tokens.get(token)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")

    def stream():
        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    app.add_api_route(
        "/stream", stream, methods=["GET"], dependencies=[Depends(login_check)]
    )
