"""Stable error envelope shared by all failure responses."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_body(code: str, message: str, request_id: str | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def error_response(
    status_code: int, code: str, message: str, request_id: str | None = None
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(code, message, request_id))
