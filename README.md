# Trace Cursor with LangSmith

Automatically send traces from [Cursor](https://cursor.com) to [LangSmith](https://smith.langchain.com).

Once configured, every Cursor agent interaction is traced — user prompts, agent responses, MCP tool calls, shell commands, file operations, and thinking steps.

## How it works

1. Global [hooks](https://cursor.com/docs/agent/hooks) run a Python script on each Cursor agent event.
2. Each message turn becomes its own trace with child runs for each tool call.
3. All turns in the same conversation are grouped into a [thread](https://docs.langchain.com/langsmith/threads) via `metadata.conversation_id`.
4. Before/after hook pairs (shell, MCP) are merged into single runs with inputs and outputs.

Tracing is **opt-in** — nothing is sent unless `TRACE_TO_LANGSMITH=true` is set. The handler is fail-open: errors never block Cursor.

## Prerequisites

- **Cursor** installed
- **Python 3.10+** with `langsmith`: `pip install langsmith`
- **LangSmith API key** — [get one here](https://smith.langchain.com/settings/apikeys)

## 1. Download the hook script

```bash
mkdir -p ~/.cursor/hooks
curl -o ~/.cursor/hooks/hook_handler.py https://raw.githubusercontent.com/langchain-ai/Cursor-LangSmith-Integration/main/hook_handler.py
```

## 2. Configure global hooks

Create or edit `~/.cursor/hooks.json`:

```json
{
  "version": 1,
  "hooks": {
    "beforeSubmitPrompt": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "afterAgentResponse": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "afterAgentThought": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "beforeShellExecution": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "afterShellExecution": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "beforeMCPExecution": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "afterMCPExecution": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "beforeReadFile": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "afterFileEdit": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "stop": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "beforeTabFileRead": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }],
    "afterTabFileEdit": [{ "command": "python3 \"$HOME/.cursor/hooks/hook_handler.py\"" }]
  }
}
```

## 3. Enable tracing

Add these environment variables to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export TRACE_TO_LANGSMITH=true
export LANGCHAIN_API_KEY=lsv2_pt_...
```

Or, to enable per-project, create a `.env` file in the project root:

```
TRACE_TO_LANGSMITH=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=my-project
```

`LANGCHAIN_PROJECT` defaults to `cursor-agent` if not set.

## 4. Verify

Send a message in Cursor and check your [LangSmith project](https://smith.langchain.com). You should see:

- Each message turn as its own trace (e.g. "Cursor: how do I create a subgraph")
- All turns in the same chat grouped into a thread
- MCP and shell calls as child runs with inputs and outputs merged
- A completion feedback score attached to each trace

## Troubleshooting

**No traces appearing?**

1. Verify environment variables are set:
   ```bash
   echo $TRACE_TO_LANGSMITH  # should be "true"
   echo $LANGCHAIN_API_KEY   # should start with "lsv2_pt_"
   ```

2. Verify `langsmith` is installed:
   ```bash
   python3 -c "import langsmith; print(langsmith.__version__)"
   ```

3. Test the hook manually:
   ```bash
   echo '{"hook_event_name":"stop","conversation_id":"test","generation_id":"test","status":"completed","loop_count":0}' \
     | TRACE_TO_LANGSMITH=true python3 ~/.cursor/hooks/hook_handler.py
   ```
   You should see `{}` on stdout with no errors.

**Cursor launched from Dock doesn't pick up env vars?**

Cursor inherits the environment of the process that launched it. If launched from the Dock (not terminal), shell profile variables won't be available. Either launch Cursor from terminal (`cursor .`) or use per-project `.env` files.
