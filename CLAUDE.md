# project_2 — mini-coding-agent + agent skills
Fork of [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent) (Ollama-backed coding agent) extended with an **agent skills** component, used to solve Lab 4 (vacuum robot Q-learning). Active feature: `intent/mini_agent_reads_skills/` (`intent.md` → `spec.md` → `plan.md`).

## Dependencies
- Python >= 3.10, [Ollama](https://ollama.com/download), default model `qwen3.5:4b`
- Runtime: stdlib only (do not add third-party runtime deps; the spec forbids it)
- Dev: `pytest`, `ruff` (see `[dependency-groups]` in `pyproject.toml`); optional `uv`
- `lab4/` has its own `requirements.txt` and `pytest.ini`

## Commands <!-- exact commands -->
```bash
ollama serve                              # separate terminal
ollama pull qwen3.5:4b
uv run mini-coding-agent --cwd .          # or: python mini_coding_agent.py
uv run mini-coding-agent --resume latest  # or a session id
uv run mini-coding-agent --help
uv run pytest -q                          # or: python -m pytest -q   (agent tests, from project_2)
cd lab4 && python -m pytest               # Lab 4 tests
uv run ruff check .
```
CLI flags: `--cwd --model --host --ollama-timeout --resume --approval {ask,auto,never} --max-steps(6) --max-new-tokens(512) --temperature(0.2) --top-p(0.9) --skill-max-chars --skill-threshold(0.25) --num-ctx(8192) --show-command-output`.
REPL commands: `/help /memory /session /reset /exit /quit /skill [<name> <prompt>]`.

## Layout
```
mini_coding_agent.py   whole agent (class MiniAgent, main); single file
tests/test_mini_coding_agent.py   agent + skills tests (FakeModelClient, build_agent helper)
skills/<name>/SKILL.md            skills (frontmatter: name, description)
lab4/                             vacuum robot: src/ (environment, q_learning, main), tests/
intent/<feature>/                 intent.md (never edit), spec.md, plan.md
```
Sessions/logs live in `.mini-coding-agent/` (gitignored): `sessions/`, `skills.jsonl`.

## Conventions
- Keep everything in `mini_coding_agent.py`; stdlib only.
- Every bug fix or feature starts as a test in `tests/test_mini_coding_agent.py`, written once.
- Follow `plan.md` step order; run the full suite after each step. Update tests only for intentionally changed behavior.
- With no skills, omit skill sections; agent-wide output and final-answer behavior applies to all requests.
- Spec requirement IDs are `REQ-n`; reference them in tests.
- Warnings go to stderr, one line each: `warning: skipped skills/<folder>/SKILL.md: <reason>`.
- Never edit `intent.md`.
- Credit: `mini_coding_agent.py`, its tests and `LICENSE` (Apache 2.0) come from upstream.

## Architecture
- Model runs in Ollama at `http://127.0.0.1:11434` (`/api/generate`); no cloud calls.
- Model must emit `<tool>...</tool>` or `<final>...</final>`; `ask` limits tool calls and permits one final-only turn after exhaustion.
- Tools: `read_file`, `write_file`, `patch_file`, `run_shell`, `delegate`, etc. Risky ones (`run_shell`, `write_file`, `patch_file`) go through `approve()`.
- Prompt = stable prefix (+ skills list) + memory + transcript; tool results retain up to 16,000 characters, and recent transcript results have a 3,000-character allowance.
- Skills: discovered from `<root>/skills/*/SKILL.md` only. Ordinary requests with skills get a separate bounded model selection pass; selected files are validated and loaded before the normal loop. `/skill` forces a skill without selection, and an empty catalog skips selection. A model may still load a skill with `read_file`. Bodies live in `active_skills` for that request only and are cleared in `ask`'s `finally`; they never enter `history`, `memory`, or session JSON (only load notices and `skills_used`).
- An unknown tool name matching a skill yields its `read_file` example without consuming a tool step; the response-attempt limit still bounds retries. Registered tools retain precedence and all permission checks.
- Selection order in `ask`: `/skill` forced -> `match_skill` similarity (cosine, `skill_similarity`, >= `--skill-threshold`) -> `select_skills` model pass only if nothing matched -> model may still `read_file` a skill. Triggers: `explicit`, `similarity`, `model-select`, `model`.
- Skill log: `log_skill_event` appends `match`/`model_select`/`executed`/`finished` JSON lines to `.mini-coding-agent/skills.jsonl` (never bodies); CLI echoes `[skills] ...` to stderr.
- Delegated child agents inherit the parent's skills list; their reads aren't recorded in the parent; they skip the similarity check.

## Verification <!-- healthy output -->
- `pytest -q`: all pass, none skipped; `lab4` tests pass separately.
- `--approval never` + risky tool: `error: approval denied`.
- Invalid skill: one stderr warning, agent still starts. No `skills/`: no output, no warning.

## Things Claude gets wrong <!-- grows from mistakes -->
- Don't test the agent with `--approval auto`.
- Tests never call a real model: use `FakeModelClient`, never real Ollama.
- Run pytest from `project_2/` for the agent and from `lab4/` for Lab 4 (separate configs); don't mix.
- Don't clear active skills per branch in `ask`; use `try/finally`. Preserve bounded tool and malformed-response limits and never execute a tool during the final-answer turn.
- Don't put skill bodies in `user_message`, history or memory; add them only in `prompt()`.
- Don't apply the ordinary tool-result cap to a skill read; the skill cap applies instead (4B: 4000, ≤13B: 12000, else 24000).
- Don't index `session["skills_used"]` directly; old sessions lack it.
- Compare resolved paths (not raw strings) when matching SKILL.md.
- Match `/skill` exactly or with a trailing space so `/skills` isn't captured.
- Line numbers in `plan.md` are stale after edits; search by symbol.
- Windows: use `\` paths in tool calls; `uv.lock` and `.mini-coding-agent` are gitignored.
