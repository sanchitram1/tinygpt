"""Application factory. Run locally with:

    uvicorn tinygpt_service.main:app --reload
"""

from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from .config import ServiceSettings
from .errors import ServiceError, error_response
from .generator import StoryGenerator, load_generator
from .limits import BodySizeLimitMiddleware, SlidingWindowRateLimiter
from .routes import STATIC_DIR, router
from .tracing import TraceSink, create_trace_sink

_UNSET = object()


def create_app(
    settings: ServiceSettings | None = None,
    generator: StoryGenerator | None | object = _UNSET,
    trace_sink: TraceSink | None = None,
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    # Fail fast: misconfigured tracing must abort startup, not surface later.
    if trace_sink is None:
        trace_sink = create_trace_sink(settings)
    load_model_on_startup = generator is _UNSET
    if load_model_on_startup:
        generator = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model_on_startup:
            # Keep model construction out of module import so test clients and
            # tooling can import the app, while real server startup remains
            # fail-fast for missing or invalid model bundles.
            app.state.generator = load_generator(app.state.settings)
        yield

    app = FastAPI(
        title="tinygpt-chat", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    app.state.settings = settings
    app.state.generator = generator
    app.state.trace_sink = trace_sink
    app.state.generation_semaphore = threading.Semaphore(1)
    app.state.rate_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return error_response(
            exc.status_code,
            exc.code,
            exc.message,
            getattr(request.state, "request_id", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = f"{location}: {first.get('msg', 'invalid request body')}"
        return error_response(
            422, "invalid_request", message, getattr(request.state, "request_id", None)
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        return error_response(
            500,
            "internal_error",
            "internal server error",
            getattr(request.state, "request_id", None),
        )

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
