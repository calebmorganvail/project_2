# Project 2 – Agent Skills for mini-coding-agent

Adds an agent skills component to [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent) and uses it to solve Lab 4 (vacuum robot Q-learning).

## Structure
```
mini_coding_agent.py   agent + skills (from rasbt/mini-coding-agent, extended)
tests/                 agent and skills tests
skills/                skill folders, each with a SKILL.md
lab4/                  Lab 4 vacuum robot (environment, Q-learning, tests)
docs/                  SDLC: intent.md, spec.md, plan.md
```

## Setup
Requires Python 3.10+ and [Ollama](https://ollama.com/download).
```bash
ollama pull qwen3.5:4b
pip install pytest
```

## Run
```bash
python mini_coding_agent.py
```

To display executed shell commands and their output as well as the final answer:
```bash
python mini_coding_agent.py --show-command-output
```
Command output can contain sensitive data. Display is off by default and does not change the approval policy.

## Output and reporting
- Tool results retain up to 16,000 characters in the session. Any discarded remainder is reported and cannot be retrieved. Session files can continue to grow, and subprocess output is captured in memory before this limit is applied.
- Recent tool results have a 3,000-character prompt allowance within a 12,000-character transcript budget. Older entries may be omitted; omitted retained content can be retrieved by the model using `read_tool_output` with its result ID and zero-based, end-exclusive character offsets (up to 2,800 characters per request). Saved older sessions are supported.
- Once the tool-call limit or malformed-response limit is reached, the agent requests one final answer without allowing further tool calls. It also makes one final attempt after a model error. If that attempt fails, the agent reports the failure; an unavailable model cannot produce a report.
- `--max-new-tokens` limits the length of the model's answer, not the input context. Skill bodies have a separate limit; a long combined prompt can still exceed the model's context capacity.

## Skills
The agent discovers skills in `<workspace>/skills/<folder>/SKILL.md` (direct subfolders only) at startup and on `/reset`.
Each file needs a frontmatter block with a single-line `name` (`[a-z0-9-]+`) and `description`:
```
---
name: train-and-report
description: Train the vacuum robot and report results.
---
Instructions...
```
- Only name, description and path go in the prompt. The model loads the body by calling `read_file` on the `SKILL.md`; the body is then active for that request only and is never saved in history, memory or the session file (the session records `skills_used`).
- **Similarity check:** every prompt is scored against each skill's name + description (cosine similarity of word counts). The best skill scoring at least `--skill-threshold` (default 0.25) is loaded automatically for that request. Below the threshold the model can still load a skill with `read_file`.
- **Skill log:** the prompt, every similarity score, the selected skill and its trigger (`similarity`, `model` or `explicit`), and the end of each skill run are appended to `.mini-coding-agent/skills.jsonl` and echoed as `[skills] ...` lines on stderr.
- `/skill` lists skills; `/skill <name> <prompt>` forces a skill for that prompt (also works as a one-shot CLI prompt).
- Delegation is unavailable while a skill is active because child agents do not inherit active skill instructions. Delegation remains available before skill activation and after the request ends. A rejected delegation still counts toward the step limit.
- Skills do not restrict tools or enforce read-only instructions. Writing, patching and shell commands remain available subject to the configured approval policy.
- Invalid skills are skipped with a one-line `warning: skipped skills/<folder>/SKILL.md: <reason>` on stderr.
- Bodies are truncated with a `[skill truncated at N chars]` notice: 4000 chars for models <=4B, 12000 for <=13B, 24000 otherwise. Override with `--skill-max-chars N`.
- Limits: multi-line and quoted frontmatter values are not supported (quotes are kept literally; `|`/`>` values are invalid); skills added mid-session appear after `/reset`; skills must be reloaded in each new request; reading a `SKILL.md` for any reason activates it until the request ends.

## Context window
`--num-ctx` (default 8192) sets Ollama's context window in tokens. Ollama silently drops the start of a prompt that is longer than its context window, and the start is where the rules, tools and skill instructions are. The character limits above do not guarantee a token budget, so raise `--num-ctx` for long sessions if your machine has the memory.

## Test
```bash
python -m pytest            # agent
cd lab4 && python -m pytest # Lab 4
```

## Credit
`mini_coding_agent.py`, `tests/test_mini_coding_agent.py` and `LICENSE` (Apache 2.0) come from rasbt/mini-coding-agent.
