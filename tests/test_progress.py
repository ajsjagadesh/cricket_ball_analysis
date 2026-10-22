from __future__ import annotations

from io import StringIO

from cricket_vision.performance.progress import ProgressReporter, format_progress


class _TerminalStream(StringIO):
    def isatty(self) -> bool:
        return True


def test_format_progress_shows_percentage_for_known_total() -> None:
    line = format_progress(current=5, total=10, elapsed_seconds=2.0, width=10)
    assert "|#####-----|  50.0% (5/10)" in line
    assert "2.5 fps" in line


def test_format_progress_shows_frame_count_without_known_total() -> None:
    line = format_progress(current=4, total=None, elapsed_seconds=2.0)
    assert "frame 4" in line
    assert "2.0 fps" in line


def test_progress_reporter_finishes_with_completion_status() -> None:
    stream = _TerminalStream()
    reporter = ProgressReporter(stream=stream, update_interval_seconds=0)
    reporter(3, 10)
    reporter.finish(succeeded=True)

    output = stream.getvalue()
    assert "Rendering video" in output
    assert "Complete" in output
    assert "30.0% (3/10)" in output
    assert output.endswith("\n")
    assert output.count("\n") == 1
    assert "\r\x1b[2K" in output


def test_progress_reporter_only_prints_final_line_when_output_is_redirected() -> None:
    stream = StringIO()
    reporter = ProgressReporter(stream=stream, update_interval_seconds=0)
    reporter(1, 10)
    reporter(2, 10)
    assert stream.getvalue() == ""

    reporter.finish(succeeded=True)
    output = stream.getvalue()
    assert output.count("\n") == 1
    assert "Complete" in output
    assert "20.0% (2/10)" in output
