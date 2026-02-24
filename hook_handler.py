#!/usr/bin/env python3
"""
Cursor → LangSmith Hook Handler

Traces Cursor AI agent activity to LangSmith. Each message turn becomes its
own trace, threaded by conversation via metadata. Before/after hook pairs
(shell, MCP) are merged into single runs using a FIFO stack on disk.

Fail-open: errors never block Cursor.

Setup:
  1. pip install langsmith
  2. Download this file to ~/.cursor/hooks/hook_handler.py
  3. Add hooks config to ~/.cursor/hooks.json (see README)
  4. Set TRACE_TO_LANGSMITH=true and LANGCHAIN_API_KEY in your environment
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

VERSION = "0.4.0"
PERMISSIVE = {"continue": True, "permission": "allow"}
STATE_DIR = Path.home() / ".cursor" / "langsmith-state"

# ---------------------------------------------------------------------------
# Stdin / env helpers
# ---------------------------------------------------------------------------

def _read_stdin() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _drain_stdin():
    try:
        sys.stdin.read()
    except OSError:
        pass


def _load_env():
    try:
        from dotenv import load_dotenv
        env = os.path.join(os.getcwd(), ".env")
        if os.path.isfile(env):
            load_dotenv(env)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# State persistence (disk-based, cross-process)
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", s)


def _state_path(namespace: str, key: str, ext: str = "state") -> Path:
    return STATE_DIR / f"{_sanitize(namespace)}_{_sanitize(key)}.{ext}"


def _save_state(namespace: str, key: str, value: str):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(namespace, key).write_text(value)


def _load_state(namespace: str, key: str) -> str | None:
    p = _state_path(namespace, key)
    return p.read_text() or None if p.exists() else None


def _push_state(namespace: str, key: str, value: str):
    p = _state_path(namespace, key, "stack")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stack: list[str] = []
    if p.exists():
        try:
            stack = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    stack.append(value)
    p.write_text(json.dumps(stack))


def _pop_state(namespace: str, key: str) -> str | None:
    p = _state_path(namespace, key, "stack")
    if not p.exists():
        return None
    try:
        stack = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not stack:
        p.unlink(missing_ok=True)
        return None
    value = stack.pop(0)
    if stack:
        p.write_text(json.dumps(stack))
    else:
        p.unlink(missing_ok=True)
    return value


def _cleanup_turn(gen_id: str):
    if not STATE_DIR.exists():
        return
    prefix = _sanitize(gen_id) + "_"
    for p in STATE_DIR.iterdir():
        if p.name.startswith(prefix):
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# LangSmith client
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        from langsmith import Client
        _client = Client()
    return _client


def _api_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        if "409" in str(e) or "Conflict" in str(e):
            return None
        raise


def _flush():
    if _client is None:
        return
    try:
        if hasattr(_client, "flush"):
            _client.flush()
        elif hasattr(_client, "tracing_queue") and _client.tracing_queue:
            _client.tracing_queue.join()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def _project() -> str:
    return os.environ.get("LANGCHAIN_PROJECT", "cursor-agent")


def _thread_meta(event: dict) -> dict:
    return {"conversation_id": event.get("conversation_id", "")}


def _tags(event: dict, extra: list[str] | None = None) -> list[str]:
    hook = event.get("hook_event_name", "")
    tags = ["cursor", "tab" if "Tab" in hook else "agent"]
    if model := event.get("model"):
        tags.append(model.replace(".", "-").lower())
    if extra:
        tags.extend(extra)
    return tags


def _ensure_root(event: dict, *, name: str = "Cursor Agent",
                 inputs: dict | None = None) -> str:
    gen_id = event.get("generation_id", "unknown")
    if existing := _load_state(gen_id, "root"):
        return existing

    root_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"cursor-langsmith:turn:{gen_id}"))
    _api_call(
        _get_client().create_run,
        name=name, inputs=inputs or {}, run_type="chain",
        id=root_id, project_name=_project(), tags=_tags(event),
        extra={"metadata": {
            **_thread_meta(event),
            "generation_id": gen_id,
            "cursor_version": event.get("cursor_version", ""),
            "model": event.get("model", ""),
            "workspace_roots": event.get("workspace_roots", []),
            "hook_handler_version": VERSION,
            "user_email": event.get("user_email", ""),
        }},
        start_time=datetime.now(timezone.utc),
    )
    _save_state(gen_id, "root", root_id)
    return root_id


def _child(event: dict, *, name: str, run_type: str = "chain",
           inputs: dict | None = None, outputs: dict | None = None,
           extra_tags: list[str] | None = None,
           extra_meta: dict | None = None) -> str:
    root_id = _ensure_root(event)
    child_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    meta: dict[str, Any] = {**_thread_meta(event), "hook_event_name": event.get("hook_event_name", "")}
    if extra_meta:
        meta.update(extra_meta)
    _api_call(
        _get_client().create_run,
        name=name, inputs=inputs or {}, run_type=run_type,
        id=child_id, parent_run_id=root_id, project_name=_project(),
        tags=_tags(event, extra_tags), extra={"metadata": meta},
        start_time=now, end_time=now, outputs=outputs,
    )
    return child_id


def _open_child(event: dict, *, name: str, inputs: dict | None = None,
                extra_tags: list[str] | None = None, state_key: str) -> str:
    root_id = _ensure_root(event)
    child_id = str(uuid.uuid4())
    gen_id = event.get("generation_id", "unknown")
    _api_call(
        _get_client().create_run,
        name=name, inputs=inputs or {}, run_type="tool",
        id=child_id, parent_run_id=root_id, project_name=_project(),
        tags=_tags(event, extra_tags),
        extra={"metadata": {**_thread_meta(event), "hook_event_name": event.get("hook_event_name", "")}},
        start_time=datetime.now(timezone.utc),
    )
    _push_state(gen_id, state_key, child_id)
    return child_id


def _close_child(event: dict, *, state_key: str,
                 outputs: dict | None = None, error: str | None = None):
    gen_id = event.get("generation_id", "unknown")
    child_id = _pop_state(gen_id, state_key)
    if not child_id:
        return
    kw: dict[str, Any] = {"end_time": datetime.now(timezone.utc)}
    if outputs:
        kw["outputs"] = outputs
    if error:
        kw["error"] = error
    _api_call(_get_client().update_run, run_id=child_id, **kw)


def _update_root(event: dict, **kw):
    gen_id = event.get("generation_id", "unknown")
    root_id = _load_state(gen_id, "root") or _ensure_root(event)
    if "end" in kw:
        kw.pop("end")
        kw["end_time"] = datetime.now(timezone.utc)
    if kw:
        _api_call(_get_client().update_run, run_id=root_id, **kw)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def _on_before_submit_prompt(e: dict) -> dict:
    prompt = e.get("prompt", "")
    attachments = e.get("attachments", [])
    _ensure_root(e, name="Cursor",
                 inputs={"prompt": prompt, "attachments": attachments})
    _child(e, name="User Prompt", run_type="llm", inputs={
        "prompt": prompt, "model": e.get("model", ""),
        "attachment_count": len(attachments),
        "attachments": [{"type": a.get("type"), "filePath": a.get("filePath")} for a in attachments],
    })
    return {"continue": True}


def _on_after_agent_response(e: dict) -> dict:
    text = e.get("text", "")
    _save_state(e.get("generation_id", "unknown"), "last_response", text)
    _child(e, name="Agent Response", run_type="llm",
           inputs={"model": e.get("model", "")}, outputs={"text": text})
    return {}


def _on_after_agent_thought(e: dict) -> dict:
    _child(e, name="Agent Thinking", run_type="chain",
           outputs={"thought": e.get("text", "")}, extra_tags=["thinking"],
           extra_meta={"duration_ms": e.get("duration_ms")} if e.get("duration_ms") else None)
    return {}


def _on_before_shell(e: dict) -> dict:
    cmd = e.get("command", "")
    _open_child(e, name=f"Shell: {cmd[:60]}",
                inputs={"command": cmd, "cwd": e.get("cwd", "")},
                extra_tags=["shell"], state_key="shell")
    return {"permission": "allow"}


def _on_after_shell(e: dict) -> dict:
    cmd, output, dur = e.get("command", ""), e.get("output", ""), e.get("duration")
    errs = _has_errors(output)
    _close_child(e, state_key="shell",
                 outputs={"output": output, "has_errors": errs, "duration_ms": dur},
                 error=f"Shell error in: {cmd[:60]}" if errs else None)
    return {}


def _on_before_mcp(e: dict) -> dict:
    tool = e.get("tool_name", "unknown")
    _open_child(e, name=f"MCP: {tool}",
                inputs={"tool_name": tool, "tool_input": e.get("tool_input", {}),
                         "server_url": e.get("url", ""), "server_command": e.get("command", "")},
                extra_tags=["mcp", f"mcp-{tool}"],
                state_key=f"mcp_{_sanitize(tool)}")
    return {"permission": "allow"}


def _on_after_mcp(e: dict) -> dict:
    tool = e.get("tool_name", "unknown")
    _close_child(e, state_key=f"mcp_{_sanitize(tool)}",
                 outputs={"result": e.get("result_json", {}), "duration_ms": e.get("duration")})
    return {}


def _on_before_read(e: dict) -> dict:
    fp = e.get("file_path", "")
    _child(e, name=f"Read: {_filename(fp)}", run_type="tool",
           inputs={"file_path": fp}, extra_tags=["file-ops"])
    return {"permission": "allow"}


def _on_after_edit(e: dict) -> dict:
    fp = e.get("file_path", "")
    stats = _edit_stats(e.get("edits", []))
    _child(e, name=f"Edit: {_filename(fp)}", run_type="tool",
           inputs={"file_path": fp}, outputs=stats, extra_tags=["file-ops"])
    return {}


def _on_before_tab_read(e: dict) -> dict:
    fp = e.get("file_path", "")
    _child(e, name=f"Tab Read: {_filename(fp)}", run_type="tool",
           inputs={"file_path": fp, "source": "tab"}, extra_tags=["file-ops"])
    return {"permission": "allow"}


def _on_after_tab_edit(e: dict) -> dict:
    fp = e.get("file_path", "")
    stats = _edit_stats(e.get("edits", []))
    _child(e, name=f"Tab Edit: {_filename(fp)}", run_type="tool",
           inputs={"file_path": fp, "source": "tab"}, outputs=stats, extra_tags=["file-ops"])
    return {}


def _on_stop(e: dict) -> dict:
    status = e.get("status", "completed")
    loops = e.get("loop_count", 0)

    gen_id = e.get("generation_id", "unknown")
    last_response = _load_state(gen_id, "last_response")
    outputs = {"response": last_response} if last_response else None

    _update_root(e, end=True, outputs=outputs,
                 error=f"Agent stopped: {status}" if status == "error" else None,
                 tags=_tags(e, [f"status-{status}"]))

    comp = {"completed": 1.0, "aborted": 0.5, "error": 0.0}.get(status, 0.5)
    root_id = _load_state(gen_id, "root")
    if root_id:
        _api_call(_get_client().create_feedback, run_id=root_id,
                  key="completion", score=comp, comment=f"{status}, {loops} loops")

    _cleanup_turn(gen_id)
    return {}


# ---------------------------------------------------------------------------
# Utilities used by handlers
# ---------------------------------------------------------------------------

def _edit_stats(edits: list[dict]) -> dict:
    added = removed = 0
    for e in edits or []:
        old = e.get("old_string") or e.get("old_line") or ""
        new = e.get("new_string") or e.get("new_line") or ""
        removed += len(old.splitlines()) if old else 0
        added += len(new.splitlines()) if new else 0
    return {"lines_added": added, "lines_removed": removed, "edit_count": len(edits or [])}


def _has_errors(output: str) -> bool:
    if not output:
        return False
    low = output.lower()
    return any(re.search(p, low) for p in [r"\berror\b", r"\bfailed\b", r"\bexception\b"])


def _filename(path: str | None) -> str:
    return path.rstrip("/").split("/")[-1] if path else "unknown"


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[[dict], dict]] = {
    "beforeSubmitPrompt": _on_before_submit_prompt,
    "afterAgentResponse": _on_after_agent_response,
    "afterAgentThought": _on_after_agent_thought,
    "beforeShellExecution": _on_before_shell,
    "afterShellExecution": _on_after_shell,
    "beforeMCPExecution": _on_before_mcp,
    "afterMCPExecution": _on_after_mcp,
    "beforeReadFile": _on_before_read,
    "afterFileEdit": _on_after_edit,
    "stop": _on_stop,
    "beforeTabFileRead": _on_before_tab_read,
    "afterTabFileEdit": _on_after_tab_edit,
}


def main():
    _load_env()

    if os.environ.get("TRACE_TO_LANGSMITH", "").lower() != "true":
        _drain_stdin()
        print(json.dumps(PERMISSIVE))
        return

    api_key = (os.environ.get("LANGCHAIN_API_KEY")
               or os.environ.get("LANGSMITH_API_KEY")
               or os.environ.get("CC_LANGSMITH_API_KEY"))
    if not api_key:
        try:
            from langsmith import Client
            Client()
        except Exception:
            _drain_stdin()
            print(json.dumps(PERMISSIVE))
            return

    try:
        event = _read_stdin()
        if not event:
            print(json.dumps(PERMISSIVE))
            return
        handler = HANDLERS.get(event.get("hook_event_name", ""))
        result = handler(event) if handler else PERMISSIVE
        _flush()
        print(json.dumps(result))
    except Exception:
        print(json.dumps(PERMISSIVE))


if __name__ == "__main__":
    main()
