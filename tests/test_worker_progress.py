"""Worker progress checks synchronize events instead of waiting 30 seconds."""
import subprocess
import threading

import pytest

from tcr_workbench import prediction


def test_heartbeat_reports_elapsed_seconds_to_stderr_only(monkeypatch, capsys):
    waits = []
    ticks = iter([False, False, True])
    times = iter([40.0, 70.0])
    class ClockEvent:
        def wait(self, interval):
            waits.append(interval)
            return next(ticks)

        def is_set(self):
            return False
    monkeypatch.setattr(prediction.time, "monotonic", lambda: next(times))
    prediction._worker_progress(ClockEvent(), 10.0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "DecoderTCR worker: 30s elapsed; still running.",
        "DecoderTCR worker: 60s elapsed; still running.",
    ]
    assert waits == [30.0, 30.0, 30.0]


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "interrupt", "oserror"])
def test_execute_stops_progress_thread_for_every_subprocess_exit(tmp_path, monkeypatch, capsys, outcome):
    entered, stopped = threading.Event(), threading.Event()
    threads = []
    def progress(stop, started):
        threads.append(threading.current_thread())
        entered.set()
        stop.wait()
        stopped.set()
    def run(command, **kwargs):
        assert entered.wait(1), "progress thread did not start"
        assert threads[0].is_alive()
        assert kwargs["stdout"] is not subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        if outcome == "oserror":
            raise OSError("worker could not start")
        kwargs["stdout"].write(b"synthetic worker diagnostics")
        return subprocess.CompletedProcess(command, 0 if outcome == "success" else 8)
    monkeypatch.setattr(prediction, "_worker_progress", progress)
    monkeypatch.setattr(prediction, "_decoder_environment", lambda executable: {})
    monkeypatch.setattr(prediction.subprocess, "run", run)
    if outcome == "success":
        prediction._execute(["synthetic-python", "worker.py"], tmp_path, 10)
    else:
        exception = {"failure": RuntimeError, "timeout": RuntimeError,
                     "interrupt": KeyboardInterrupt, "oserror": OSError}[outcome]
        with pytest.raises(exception) as captured:
            prediction._execute(["synthetic-python", "worker.py"], tmp_path, 10)
        if outcome == "failure":
            assert "exit 8" in str(captured.value) and "synthetic worker diagnostics" in str(captured.value)
        elif outcome == "timeout":
            assert "timeout=10s" in str(captured.value)
    assert stopped.is_set() and len(threads) == 1 and not threads[0].is_alive()
    assert threads[0].name == "DecoderTCR-progress"
    assert capsys.readouterr().out == ""


def test_fast_worker_finishes_without_progress_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prediction, "_decoder_environment", lambda executable: {})
    monkeypatch.setattr(prediction.subprocess, "run",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    prediction._execute(["synthetic-python"], tmp_path, None)
    assert capsys.readouterr() == ("", "")


def test_interrupt_during_progress_start_sets_stop_without_joining_unstarted_thread(tmp_path, monkeypatch):
    events = []
    class InterruptedThread:
        ident = None

        def __init__(self, *, args, **kwargs):
            events.append(args[0])

        def start(self):
            raise KeyboardInterrupt()

        def join(self):
            pytest.fail("must not join a thread that never started")
    monkeypatch.setattr(prediction.threading, "Thread", InterruptedThread)
    monkeypatch.setattr(prediction.subprocess, "run", lambda *a, **k: pytest.fail("worker started after interruption"))
    with pytest.raises(KeyboardInterrupt):
        prediction._execute(["synthetic-python"], tmp_path, None)
    assert len(events) == 1 and events[0].is_set()


def test_closed_stderr_does_not_break_heartbeat(monkeypatch):
    class ReadyEvent:
        def wait(self, interval):
            return False

        def is_set(self):
            return False
    class ClosedStream:
        def write(self, message):
            raise ValueError("stream closed")
    monkeypatch.setattr(prediction.sys, "stderr", ClosedStream())
    prediction._worker_progress(ReadyEvent(), 0.0)


def test_missing_stderr_never_redirects_progress_into_stdout(monkeypatch, capsys):
    class ReadyEvent:
        def wait(self, interval):
            return False

        def is_set(self):
            return False
    monkeypatch.setattr(prediction.sys, "stderr", None)
    prediction._worker_progress(ReadyEvent(), 0.0)
    assert capsys.readouterr().out == ""
