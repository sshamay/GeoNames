"""OpenTelemetry tracing for the GeoNames agent.

Provides a thin, optional OTel wrapper: when an OTLP endpoint is configured an
exporter is started and the agent can emit spans for its tool calls; otherwise
a no-op tracer is used so the suite stays offline and dependency-free at test
time. Nothing here is required for the AQuA golden-anchor contract - that runs
off the in-memory ``TestTraceCollector`` - this is the deep-telemetry channel
(AQuA observe pillar) layered on top.
"""

from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracer: Optional[trace.Tracer] = None
_initialized = False
_export_enabled = False

# Env knob: OTEL_EXPORTER_OTLP_ENDPOINT or a GeoNames-specific monotonic default.
_OTLP_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"


def _install_exporter(endpoint: str) -> None:
    """Install the OTLP HTTP span processor for the given endpoint."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except Exception:  # pragma: no cover - exporter optional
        return
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )


def init_telemetry(
    service_name: str = "geonames-agent",
    *,
    endpoint: Optional[str] = None,
    enable: Optional[bool] = None,
) -> None:
    """Initialise the OTel tracer provider (idempotent).

    Args:
        service_name: Service name for the OTel resource.
        endpoint: OTLP HTTP endpoint. If None, read ``OTEL_EXPORTER_OTLP_ENDPOINT``.
        enable: Force on/off. Defaults to ``True`` if an endpoint is available.

    The provider is always created (so ``trace.get_tracer`` is safe), but spans
    are only exported to OTLP when an endpoint is configured.
    """
    global _tracer, _initialized
    if _initialized:
        return
    _initialized = True

    endpoint = endpoint or os.environ.get(_OTLP_ENDPOINT_ENV)
    should_export = endpoint and (enable if enable is not None else True)

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    trace.set_tracer_provider(provider)
    if should_export:
        _install_exporter(endpoint)
        global _export_enabled
        _export_enabled = True
    _tracer = trace.get_tracer(service_name)


def get_tracer(service_name: str = "geonames-agent") -> trace.Tracer:
    """Return the provider tracer, initialising lazily if needed."""
    global _tracer
    if _tracer is None:
        init_telemetry(service_name, enable=False)
    return _tracer or trace.get_tracer(service_name)


def is_enabled() -> bool:
    """Whether OTLP export is active (vs. a no-op tracer)."""
    return _export_enabled
