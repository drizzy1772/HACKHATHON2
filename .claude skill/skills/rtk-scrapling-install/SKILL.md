---
name: rtk-scrapling-install
description: Install and wire up two Claude Code tools — RTK (token-optimizing 
  CLI proxy) and Scrapling (web scraping library + MCP server). Idempotent. Each
  step checks if the artifact is already present and skips it; otherwise 
  installs. Prints a summary of what was newly installed vs already present.
---

# rtk-scrapling-install

You are installing and registering two tools for use with Claude Code in this repo:

1. **RTK** ([rtk-ai/rtk](https://github.com/rtk-ai/rtk)) — a CLI proxy that filters/compresses command output to reduce LLM token usage. Two layers: a project-local docs install (RTK instructions appended to `CLAUDE.md`) and a global PreToolUse hook (`rtk hook claude` in `~/.claude/settings.json`) that auto-rewrites `Bash` commands through `rtk`.
2. **Scrapling** ([D4Vinci/Scrapling](https://github.com/D4Vinci/Scrapling)) — a Python web-scraping framework. Installed with `ai` and `fetchers` extras so it exposes an MCP server via `scrapling mcp`, registered in this repo's [.mcp.json](../../../.mcp.json).

## Hard rules

- **Idempotent.** Each step checks first. Don't re-run an install that's already done.
- **No new prompts.** The user invoked the skill expecting the install to happen. Don't re-ask scope decisions that are already baked in below.
- **Don't bypass safety.** If a step needs sudo or a destructive action, stop and report — don't escalate silently.
- **Verify after each step.** If a verification fails, halt and report; don't move to the next step.
- **Project-local RTK init modifies committed files** ([CLAUDE.md](../../../CLAUDE.md), creates [.rtk/filters.toml](../../../.rtk/), and [.mcp.json](../../../.mcp.json)). Mention this at the end so the user knows what `git status` will show.

## Steps

Run these in order. Each has a "Skip if" guard — check it first; if it matches, log "already installed" and move on.

### 1. RTK binary

- **Skip if:** `command -v rtk` succeeds AND `rtk --version` reports any version.
- **Install:** `curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh`
  - Installs to `~/.local/bin/rtk`. Assumes `~/.local/bin` is in `PATH` (it is on this machine — verified at skill creation).
- **Verify:** `rtk --version` returns `rtk <semver>`.

### 2. RTK project-local docs

- **Skip if:** `grep -q '<!-- rtk-instructions' CLAUDE.md` succeeds (the marker comment is present).
- **Install:** `rtk init --auto-patch` from the repo root.
  - Appends ~140 lines of RTK usage docs to [CLAUDE.md](../../../CLAUDE.md).
  - Creates [.rtk/filters.toml](../../../.rtk/filters.toml) (template; user edits later).
  - Does NOT install a hook (project-local mode prints a warning about this — that's expected; the global hook in step 3 handles it).
- **Verify:** Both `grep -q '<!-- rtk-instructions' CLAUDE.md` and `test -f .rtk/filters.toml` succeed.

### 3. RTK global PreToolUse hook

- **Skip if:** `python3 -c "import json,sys; d=json.load(open('$HOME/.claude/settings.json')); sys.exit(0 if any(h.get('command')=='rtk hook claude' for m in d.get('hooks',{}).get('PreToolUse',[]) for h in m.get('hooks',[])) else 1)"` succeeds.
- **Install:** `rtk init -g --auto-patch`.
  - Patches `~/.claude/settings.json` with a `PreToolUse`→`Bash` hook running `rtk hook claude` (auto-rewrites Bash commands through RTK).
  - Writes `~/.claude/RTK.md` and references it from `~/.claude/CLAUDE.md`.
  - Backs up the previous `settings.json` to `~/.claude/settings.json.bak`.
- **Verify:** The skip-if check above now passes.
- **Note:** The hook only activates after Claude Code is restarted. Mention this in the final summary.

### 4. Scrapling pip install

- **Skip if:** `python3 -c "import scrapling" 2>/dev/null` succeeds AND `python3 -c "import mcp" 2>/dev/null` succeeds (confirms the `[ai]` extra is present).
- **Prereqs:** Python ≥ 3.10. Verify with `python3 --version`.
- **Install:** `pip install "scrapling[ai,fetchers]"` (system pip — this repo has no venv).
- **Verify:** `scrapling --help` lists the `mcp` subcommand; `python3 -c "import scrapling, mcp"` runs clean.

### 5. Scrapling browsers

- **Skip if:** `ls ~/.cache/ms-playwright/chromium-*/chrome-linux/chrome 2>/dev/null` returns a path.
- **Install:** `scrapling install`. This downloads Chromium + Playwright deps and runs `apt` for system libs — can take 2–5 minutes. Use a long timeout on the Bash call.
- **Verify:** The skip-if check above now passes.

### 6. Register Scrapling MCP in [.mcp.json](../../../.mcp.json)

- **Skip if:** `python3 -c "import json,sys; sys.exit(0 if 'scrapling' in json.load(open('.mcp.json')).get('mcpServers',{}) else 1)" 2>/dev/null` succeeds.
- **Install:** Create or merge into [.mcp.json](../../../.mcp.json) at the repo root:
  ```json
  {
    "mcpServers": {
      "scrapling": {
        "command": "scrapling",
        "args": ["mcp"]
      }
    }
  }
  ```
  - If `.mcp.json` already exists with other `mcpServers`, merge the `scrapling` entry in; don't clobber the file. Use the `Read` then `Edit` tools, not `Write`.
- **Verify:** The skip-if check above now passes.
- **Note:** Scrapling MCP only activates after Claude Code is restarted and the user approves the project-scope MCP server on first load (Claude Code prompts on untrusted `.mcp.json` changes).

## Final summary

After all steps, print a concise report in this format:

```
RTK
- binary:        <newly installed | already present at ~/.local/bin/rtk> (vX.Y.Z)
- project docs:  <newly installed | already present> (CLAUDE.md, .rtk/filters.toml)
- global hook:   <newly installed | already present> (~/.claude/settings.json)

Scrapling
- pip package:   <newly installed | already present> (scrapling X.Y.Z, with [ai,fetchers])
- browsers:      <newly installed | already present> (~/.cache/ms-playwright/)
- MCP entry:     <newly added | already present> (.mcp.json → scrapling)

Next steps
- Restart Claude Code so the RTK hook and Scrapling MCP server are loaded.
- On next start, approve the new MCP server when prompted.
- Modified/created files (git status will show these):
  <list only the files this run actually changed>
```

## Failure handling

- If `pip install` fails because of a system Python lockdown (PEP 668 `externally-managed-environment`), stop and tell the user — don't `--break-system-packages` without explicit confirmation. Suggest creating a venv as the alternative.
- If `rtk init` fails or `rtk` isn't on PATH after step 1, check `echo $PATH` for `~/.local/bin`. If missing, halt — don't try to edit shell rc files without asking.
- If a `.mcp.json` merge would lose existing entries, halt and show the user the conflict.
