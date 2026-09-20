# Use an assistant with TCR-Workbench

An assistant can help prepare input tables, choose a command and inspect results. All workflows also run directly through `python3 tcr.py`.

## Start a session

Open the repository folder in your installed assistant. Ask it to read `AGENTS.md` and the [workflow skill](../skills/tcr-workbench/SKILL.md) before starting an analysis.

| Client | Project instructions |
|---|---|
| Codex | [AGENTS.md](../AGENTS.md) |
| Claude Code | [CLAUDE.md](../CLAUDE.md) |
| Gemini CLI | [GEMINI.md](../GEMINI.md) |
| Antigravity / `agy` | [AGENTS.md](../AGENTS.md) |

For example:

> Read AGENTS.md and skills/tcr-workbench/SKILL.md. Check my installation with doctor, then help me screen my processed 10x receptors against my peptide panel using donor HLA typing. Tell me which files you need. Write results to a new folder and explain any unresolved inputs.

The model runs locally, but your assistant may send inspected files to a remote service. Use your lab's approved settings for private data.
