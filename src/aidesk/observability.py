"""OpenTelemetry tracing. Spans go to var/traces.jsonl, and to any OTLP collector you configure
(for example a local Arize Phoenix or Langfuse) when OTEL_EXPORTER_OTLP_ENDPOINT is set."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from aidesk.config import get_settings

_configured = False
_lock = threading.Lock()


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = []
        for s in spans:
            ctx = s.get_span_context()
            lines.append(json.dumps({
                "name": s.name,
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                "start": s.start_time,
                "duration_ms": round(((s.end_time or s.start_time) - s.start_time) / 1e6, 1),
                "status": s.status.status_code.name,
                "attributes": dict(s.attributes or {}),
            }, default=str))
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS


def setup_tracing(service: str = "aidesk") -> None:
    global _configured
    with _lock:
        if _configured:
            return
        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter(get_settings().data_dir / "traces.jsonl")))
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            except ImportError:
                pass
        trace.set_tracer_provider(provider)
        _configured = True


def recent_traces(limit: int = 20) -> list[dict]:
    path = get_settings().data_dir / "traces.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(line) for line in lines if line.strip()]
