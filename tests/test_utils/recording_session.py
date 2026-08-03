"""HTTP session that records every request for assertion in tests.

Lets user-flow tests prove transport-level facts (HTTP 200 responses, the
exact query params sent) without losing the raw response body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class RecordedCall:
    """One captured HTTP request."""

    method: str
    url: str
    params: Optional[Dict[str, Any]]
    status_code: int


class RecordingSession(requests.Session):
    """A requests.Session that records each request it makes."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[RecordedCall] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        response = super().request(method, url, **kwargs)
        self.calls.append(
            RecordedCall(
                method=method,
                url=url,
                params=kwargs.get("params"),
                status_code=response.status_code,
            )
        )
        return response

    @property
    def status_codes(self) -> List[int]:
        """HTTP status of every recorded call, in order."""
        return [call.status_code for call in self.calls]

    def find_calls(self, endpoint: str) -> List[RecordedCall]:
        """Recorded calls whose URL targets ``endpoint`` (e.g. ``earthquakesJSON``)."""
        return [
            call for call in self.calls if call.url.rstrip("/").endswith(endpoint)
        ]
