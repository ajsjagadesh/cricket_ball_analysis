from __future__ import annotations

from cricket_vision.performance.metrics import MetricsCollector


def test_metrics_collector_reports_frame_rate_and_stage_aggregates() -> None:
    metrics = MetricsCollector()
    metrics.start()
    metrics.record("decode", 0.01)
    metrics.record("decode", 0.02)
    metrics.frame_processed()
    metrics.frame_processed()

    summary = metrics.summary()
    assert summary["processed_frames"] == 2
    assert summary["processing_fps"] > 0
    assert summary["stages"]["decode"]["calls"] == 2
    assert summary["stages"]["decode"]["mean_seconds"] == 0.015
