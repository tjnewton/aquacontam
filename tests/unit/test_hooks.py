"""Contract tests for the Claude Code hooks in ``.claude/hooks/``.

These guard three defect classes that together left six of the seven hooks
silently non-functional (see PROGRESS.md, 2026-08-04):

1. A bare top-level ``{"permissionDecision": ...}``, which Claude Code ignores,
   so both safety guards denied nothing even when their scripts ran.
2. Reasons built by string-interpolating text that contains quotes, backslashes
   or newlines, producing invalid JSON that is silently dropped.
3. Failing OPEN when the payload cannot be parsed - a guard that cannot
   evaluate its rules must never behave like one that passed.

Only the four side-effect-free hooks are exercised. ``session-start.sh``,
``stop-progress-tracker.sh`` and ``stop-quality-gate.sh`` are deliberately never
invoked here: they write into ``.claude/`` and ``paper/``, and the quality gate
spawns a full nested pytest run.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
TIMEOUT_S = 30

if not HOOKS_DIR.is_dir():  # export-ignored, so absent in the released archive
    pytest.skip(".claude/hooks/ is not present", allow_module_level=True)


def _find_working_bash() -> str | None:
    """Locate a bash that actually runs.

    ``shutil.which("bash")`` is not sufficient on Windows: it resolves to
    ``C:\\WINDOWS\\system32\\bash.EXE``, the WSL launcher, which exits non-zero
    when no distribution is installed. Probe each candidate instead of trusting
    its existence.
    """
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "echo ok"], capture_output=True, text=True, timeout=TIMEOUT_S
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0 and "ok" in probe.stdout:
            return candidate
    return None


BASH = _find_working_bash()
if BASH is None:
    pytest.skip("no working bash interpreter found", allow_module_level=True)

GUARDS = ("pre-bash-guard.sh", "pre-write-protect.sh")
BLOCKED_COMMAND = "rm -r" + "f /"  # split so this file never contains the literal


def run_hook(
    name: str, payload: dict[str, object], env: dict[str, str] | None = None
) -> tuple[int, str]:
    """Run hook ``name`` with ``payload`` on stdin; return (returncode, stdout)."""
    proc = subprocess.run(
        [BASH, str(HOOKS_DIR / name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=env,
    )
    return proc.returncode, proc.stdout


def deny_payload(stdout: str) -> dict[str, str]:
    """Parse a PreToolUse deny decision and assert the nested shape."""
    hook_out = json.loads(stdout)["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "PreToolUse", hook_out
    assert hook_out["permissionDecision"] == "deny", hook_out
    assert hook_out["permissionDecisionReason"], hook_out
    return hook_out


# --------------------------------------------------------------------------
# Static checks. Written as helpers over supplied text so each one is applied
# twice: to the real hooks (must pass) and to a synthetic bad string (must
# flag), proving the check can actually fail.
# --------------------------------------------------------------------------
def bare_permission_decision(text: str) -> bool:
    """True if a decision object opens with permissionDecision (the old defect).

    Comment lines are skipped: the hooks document the old broken shape in their
    own headers, and flagging that documentation would be a false positive.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if '{"permissionDecision"' in stripped.replace(" ", ""):
            return True
    return False


def bare_python3_lines(text: str) -> list[str]:
    """Executable ``python3`` invocations outside a resolver candidate list."""
    hits = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "_cand in python python3" in stripped:
            continue
        if "python3" in stripped:
            hits.append(stripped)
    return hits


@pytest.mark.parametrize("hook", sorted(p.name for p in HOOKS_DIR.glob("*.sh")))
def test_no_bare_permission_decision(hook: str) -> None:
    text = (HOOKS_DIR / hook).read_text(encoding="utf-8")
    assert not bare_permission_decision(text), (
        f"{hook} emits a top-level permissionDecision, which Claude Code ignores; "
        "nest it under hookSpecificOutput"
    )


def test_bare_permission_decision_detector_can_fail() -> None:
    """Negative control: the checker must flag the shape it exists to catch."""
    assert bare_permission_decision('{"permissionDecision":"deny","reason":"x"}')
    nested = '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny"}}'
    assert not bare_permission_decision(nested)


@pytest.mark.parametrize("hook", sorted(p.name for p in HOOKS_DIR.glob("*.sh")))
def test_no_bare_python3_invocation(hook: str) -> None:
    text = (HOOKS_DIR / hook).read_text(encoding="utf-8")
    assert bare_python3_lines(text) == [], (
        f"{hook} invokes python3 directly; Windows/conda ships `python` only, "
        "so it must go through the resolver"
    )


def test_bare_python3_detector_can_fail() -> None:
    """Negative control for the interpreter check."""
    assert bare_python3_lines('X="$(echo "$I" | python3 -c "import json")"')
    assert bare_python3_lines("# a comment about python3") == []
    assert bare_python3_lines("for _cand in python python3; do :; done") == []


# --------------------------------------------------------------------------
# Behavioural checks against the four side-effect-free hooks.
# --------------------------------------------------------------------------
def test_bash_guard_denies_destructive_command() -> None:
    rc, out = run_hook("pre-bash-guard.sh", {"tool_input": {"command": BLOCKED_COMMAND}})
    assert rc == 0
    assert "destructive" in deny_payload(out)["permissionDecisionReason"]


def test_bash_guard_allows_benign_command() -> None:
    rc, out = run_hook("pre-bash-guard.sh", {"tool_input": {"command": "git status"}})
    assert (rc, out.strip()) == (0, "")


@pytest.mark.parametrize(
    "name", [".env", ".env.local", "credentials.json", "secrets.json", "id.pem", "server.key"]
)
def test_write_guard_denies_protected_basenames(name: str) -> None:
    rc, out = run_hook("pre-write-protect.sh", {"tool_input": {"file_path": f"/tmp/{name}"}})
    assert rc == 0
    deny_payload(out)


@pytest.mark.parametrize("rel", ["data/raw/x.csv", ".git/config"])
def test_write_guard_denies_protected_paths(rel: str) -> None:
    """Path rules match by string against PROJECT_DIR - no file is created."""
    rc, out = run_hook("pre-write-protect.sh", {"tool_input": {"file_path": str(REPO_ROOT / rel)}})
    assert rc == 0
    deny_payload(out)


def test_write_guard_allows_ordinary_source_path() -> None:
    payload = {"tool_input": {"file_path": str(REPO_ROOT / "src" / "aquacontam" / "x.py")}}
    rc, out = run_hook("pre-write-protect.sh", payload)
    assert (rc, out.strip()) == (0, "")


@pytest.mark.parametrize("hook", GUARDS)
def test_guards_fail_closed_on_unparseable_payload(hook: str) -> None:
    """A guard that cannot evaluate its rules must deny, never allow."""
    proc = subprocess.run(
        [BASH, str(HOOKS_DIR / hook)],
        input="this is not json",
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert proc.returncode == 0
    deny_payload(proc.stdout)


def test_deny_reason_escapes_quotes_and_backslashes() -> None:
    """Reasons embed user text, so they must be JSON-encoded, not interpolated.

    Runs without ruff, so it is the CI-side guard for this defect class. Two
    constraints on the fixture: the basename must START with a protected prefix
    (``.env*``) or the hook correctly allows it, and it must not contain a
    backslash, which ``basename`` treats as a separator on Windows and would
    truncate away the interesting part. A bare double quote is enough to break
    naive interpolation.
    """
    nasty = '.env"quote\tand\ttabs'
    rc, out = run_hook("pre-write-protect.sh", {"tool_input": {"file_path": f"/tmp/{nasty}"}})
    assert rc == 0
    reason = deny_payload(out)["permissionDecisionReason"]
    assert '"' in reason, reason
    assert "\t" in reason, reason


@pytest.mark.parametrize("hook", GUARDS)
def test_guards_fall_back_when_interpreter_is_broken(hook: str, tmp_path: Path) -> None:
    """With every interpreter candidate failing, a static deny is still emitted."""
    bin_dir = tmp_path / "decoy"
    bin_dir.mkdir()
    for name in ("python", "python3", "py"):
        decoy = bin_dir / name
        decoy.write_text("#!/bin/sh\nexit 9\n")
        decoy.chmod(decoy.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    probe = subprocess.run(
        ["bash", "-c", "command -v python && python -c 'pass'"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=env,
    )
    if probe.returncode == 0:
        pytest.skip("decoy interpreter was not picked up by this shell")

    rc, out = run_hook(
        hook, {"tool_input": {"command": "echo hi", "file_path": "/tmp/x"}}, env=env
    )
    assert rc == 0
    deny_payload(out)


# --------------------------------------------------------------------------
# ruff-dependent. The CI `test` job installs .[test], which has no ruff, so
# these skip there; test_deny_reason_escapes_quotes_and_backslashes above is
# what covers the invalid-JSON defect remotely.
# --------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_lint_hook_emits_valid_json_for_violations(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text('import os\nimport sys\nX = "a"\n')
    rc, out = run_hook("post-edit-lint.sh", {"tool_input": {"file_path": str(bad)}})
    assert rc == 0
    data = json.loads(out)  # the defect produced JSONDecodeError here
    assert data["decision"] == "block"
    assert "\n" in data["reason"], "multi-line ruff output should survive encoding"


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_lint_hook_is_quiet_for_clean_file(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    good.write_text("X = 1\n")
    rc, out = run_hook("post-edit-lint.sh", {"tool_input": {"file_path": str(good)}})
    assert rc == 0
    assert json.loads(out) == {"suppressOutput": True}
