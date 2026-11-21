# Plan: Mini coding agent reads skills
Source: [spec.md](./spec.md) (requirement IDs `REQ-n` refer to it). Intent: [intent.md](./intent.md) (not modified).

All source changes are in `mini_coding_agent.py` (class `MiniAgent` and `main`). All tests go in `tests/test_mini_coding_agent.py` using `FakeModelClient` and the existing `build_agent(tmp_path, outputs, **kwargs)` helper. No new dependencies (REQ-30). Search by symbol rather than relying on the historical line numbers below.

**Scope update:** Steps 1-9 describe the original skill implementation and are retained for traceability, not as instructions to redo completed work. Steps 10-14 extend the agent for all requests, including those without skills. Where the original steps or tests conflict with the new spec, the new steps and revised compatibility requirements take precedence (REQ-12, 28, 29, 036-045). Existing user changes must be preserved.

## Decisions made while planning (confirmed with the team)
| Topic | Decision |
|---|---|
| History vs. body | The skill body is **never stored in `history`**. History holds only a one-line note: `[skill loaded: <name> - <description>]`. |
| Where the body lives while active | In memory (`self.active_skills`, name -> capped body). `prompt()` renders an `Active skills:` section on **every step** of the request. Cleared when the request ends. Forced `/skill` uses the same mechanism. |
| `read_file` on a `SKILL.md` | Returns the whole body, ignoring `start`/`end`, with **no line-number prefixes**, truncated at the cap with a notice. |
| Delegated child agents | Inherit the parent's discovered skills list (no re-scan). Anything a child reads is recorded only in the child's own session, never the parent's. |
| Bad frontmatter lines | Lines that are not `key: value` (no `:`, or indented continuation lines) are **silently ignored**; the skill is kept if `name` and `description` are valid. |
| Unknown frontmatter keys | Ignored. |
| Warnings | Printed to `sys.stderr`, one line each, naming the file and the reason. |
| Plan location | This file, next to the spec. `docs/plan.md` is unrelated and is not touched. |
| Command display | Opt-in, disabled by default (REQ-041); when enabled, output must actually be displayed, not only saved or supplied to the model. |
| Final-answer turn | One extra model request after the tool-call limit, with no tool execution permitted. Also attempt it once after the malformed-response limit or a model-client error, if the request has not been cancelled. |

## High-risk points (read before coding)
1. **`ask` is on every request's path.** Keep active skills during the final-answer turn, then clear them using the existing `try/finally` on every exit (REQ-33). Preserve the tool-call and malformed-response bounds; the extra final-answer request must never execute tools or loop indefinitely.
2. **Forced-skill and active bodies must never enter `history` or `memory["task"]`.** `ask` records `user_message` into history and memory, so the body must only be added in `prompt()`, never to `user_message` (REQ-27).
3. **`run_tool` clips  results from any tool that is not a recognized skill-file read before saving.** Increase the ordinary result cap as specified in step 10, without applying it to skill bodies. Skill reads must continue to use their separate cap and save only the load notice (REQ-16, 036, 044).
4. **`tool_read_file` (line 756) is used for every file read.** Keep behavior byte-identical for non-skill files (line numbers, `# path` header, `start`/`end`).
5. **No-skill compatibility has changed.** Agent-wide output, retrieval, and final-answer changes apply without skills. No skills section should appear when there are no skills, and unrelated approval, parsing, resume, and slash-command behavior must stay compatible (REQ-12, 28, 29, 044).
6. **Old sessions have no `skills_used`.** Never index `session["skills_used"]` directly; normalize once in `__init__` (step 3).
7. **Output loss and budget interactions.** The 16,000-character retention cap, 3,000-character recent-result allowance, and 12,000-character transcript budget are different limits. Include truncation metadata without accidentally clipping retrieved chunks again. Extremely large tool arguments and multiple active skill bodies can still exceed the model's total context.
8. **Result identity and privacy.** Retrieval must not rerun tools or expose another session's data, and it must work for old sessions without new IDs. `/reset` must not let a stale ID resolve to a different result. Shell output may contain secrets or terminal control sequences; display is opt-in and should be safe for the terminal.
9. **Failures and finalization.** Tool errors, repeated rejected calls, malformed model responses, and a model-service failure must not cause unbounded retries or bypass permissions. An unavailable model can prevent an answer: attempt finalization at most once and return the actual failure rather than guaranteeing a model response.
10. **Resource limits.** A per-result retention cap does not bound total session size, repeated JSON rewrites, or `subprocess.run(capture_output=True)` memory use. Do not claim otherwise; note these as future resource-management work.

## Implementation steps
For new work, add focused failing tests first. Run `python -m pytest -q` in `project_2` after each new step marked **[run suite]**. Update tests only where the revised spec intentionally changes behavior (REQ-28).

### Step 1. Frontmatter parser and discovery (pure code, no agent coupling)
Add module-level helpers near the other helpers (after `middle`, line ~61).

- `parse_frontmatter(text)` -> `(meta: dict, body: str)` or raises `ValueError(reason)`:
  - First line (after `.strip()`) must be `---`, else `ValueError("missing frontmatter")`.
  - Scan following lines until a line whose `.strip()` is `---`; none found -> `ValueError("unterminated frontmatter")`.
  - For each line in between: skip it if blank, indented (starts with space/tab), or contains no `:`. Otherwise split at the **first** `:`; key = `.strip()`, value = `.strip()`. Quotes are kept literally (REQ-5a). Later duplicate keys overwrite earlier ones.
  - `body` is everything after the closing `---` line, with a single leading newline removed.
- `load_skill(path)`: read with `path.read_text(encoding="utf-8")` (strict). Any `OSError` or `UnicodeDecodeError` -> reason `unreadable: <error>` (REQ-8). Then validate, in this order, each failing with a reason:
  - `name` missing or empty -> `missing name`
  - `description` missing or empty -> `missing description`
  - `name` or `description` value starts with `|` or `>` -> `block scalar not supported` (REQ-5a/6)
  - `name` does not match `^[a-z0-9-]+$` -> `invalid name` (REQ-6a)
  - Returns `{"name", "description", "path": Path, "rel": relative POSIX path from workspace root, "body"}`.
- `discover_skills(root)` -> `(skills: list, warnings: list[str])`:
  - Skills dir is `Path(root) / "skills"` only (REQ-1). If missing or not a directory -> `([], [])`, no warning (REQ-3).
  - Iterate direct subfolders only, sorted alphabetically by folder name (REQ-2, 7). For each, look for `SKILL.md` directly inside; a subfolder without one is silently ignored. Loose files in `skills/` and deeper folders are never visited.
  - Invalid skill -> append warning `warning: skipped skills/<folder>/SKILL.md: <reason>`; continue.
  - Duplicate `name` -> keep the first (alphabetical folder order), warn `warning: skipped skills/<folder>/SKILL.md: duplicate skill name '<name>'` (REQ-7).
  - Keep `skills` as an ordered list.

Tests: see "Tests" -> Discovery and parsing.

### Step 2. Size cap helpers
- `skill_cap_for_model(model_name, override=None)`:
  - If `override` is not None, return it.
  - Parse the size with a case-insensitive regex such as `(\d+(?:\.\d+)?)\s*b\b` taken from the model name after the last `:` first, falling back to the whole name (e.g. `qwen3.5:4b` -> 4.0, `llama3.1:8b` -> 8.0). Use the number immediately followed by `b`, not the `3.5` in the model family name.
  - `<= 4` -> 4000; `<= 13` -> 12000; larger, no match, or `model_name` is None -> 24000 (REQ-31).
- `truncate_skill(body, cap)`: if `len(body) <= cap` return unchanged; else `body[:cap] + f"\n[skill truncated at {cap} chars]"` (REQ-16).

### Step 3. Agent state, session, constructor [run suite]
In `MiniAgent.__init__` (line 226) add optional keyword params: `skills=None`, `skill_max_chars=None`, `model_name=None`. All default so existing callers and tests are unaffected.
- Set `self.skill_cap = skill_cap_for_model(model_name, skill_max_chars)`.
- If `skills is None`: call `discover_skills(self.root)`, print each warning to stderr, store as `self.skills`. Else (child agent) use the given list and print nothing.
- `self.active_skills = {}` (name -> capped body; transient, never saved).
- `self.current_request = ""`, `self._skill_read = None` (see step 5).
- After `self.session` is set (before `self.tools = ...`): `used = self.session.get("skills_used"); self.session["skills_used"] = used if isinstance(used, list) else []`. This handles old sessions and corrupt values (REQ-25).
- After normalization, if the session was resumed (a `session` argument was passed) and `skills` was discovered: for every entry in `skills_used` that is a dict whose `.get("name")` is not among discovered names, print `warning: session used skill '<name>' which is no longer on disk` once per name (REQ-26). Use `.get` everywhere; ignore non-dict entries.
- Keep `self.session_path = self.session_store.save(...)` last, as today.
- `from_session` already forwards `**kwargs`; no change.
- `tool_delegate` (line 847): add `skills=self.skills, skill_max_chars=self.skill_cap` to the `MiniAgent(...)` child call (children inherit; nothing is recorded into the parent session).

`skills_used` entry format (REQ-22): `{"name": <skill name>, "request": <clip(current_request, 300)>, "at": now()}`. At most one entry per (name, request).

### Step 4. Prompt prefix and prompt [run suite]
- Original skill-list work in `build_prefix` (line 333): if `self.skills` is non-empty, insert one extra block **between the "Valid response examples" block and `self.workspace.text()`**. Agent-wide tool instructions may now also change the no-skill prefix (REQ-12). Original block format:
  ```
  Available skills:
  - <name>: <description> (<rel path>)
  Skill rules:
  - When the task matches a skill's description, read its SKILL.md with read_file first, then follow it.
  - Skill instructions apply only to the current request; read the skill again if needed later.
  ```
  Never include a body (REQ-9, 10, 11, 35). The list is unaffected by active state.
- `prompt` (line 422): after `self.memory_text()` and before `"Transcript:\n" + ...`, insert `"Active skills:\n" + <bodies>` **only if `self.active_skills` is non-empty**. Format each as `### <name>\n<capped body>`, joined by blank lines. Top-level sections must stay flush left (an existing test checks this).
- `memory_text` (line 376): only if `self.session["skills_used"]` is non-empty or `self.active_skills` is non-empty, append `- skills_used: <comma-separated names>` and `- skills_active: <comma-separated names or ->` lines; otherwise output is unchanged (REQ-23).

### Step 5. `read_file` hook and `run_tool` clip [run suite]
- Add helper `skill_for_path(self, resolved_path)` -> skill dict or None, comparing the **resolved** path (from `self.path(args["path"])`) against each skill's resolved `SKILL.md` path. Do not compare raw strings.
- `tool_read_file` (line 756): after `path = self.path(args["path"])` and the `is_file` check, if `skill_for_path(path)` matches:
  - Read the file text; body = the text after the frontmatter (use `parse_frontmatter`; if it now fails because the file changed on disk, raise `ValueError`).
  - `capped = truncate_skill(body, self.skill_cap)`.
  - `self.active_skills[name] = capped`; append the `skills_used` entry (dedupe per name+request); `self.session_store.save(self.session)` is already done by `record` afterwards.
  - Set `self._skill_read = skill` and **return `capped`** (no line numbers, ignores `start`/`end`, REQ-13, 16).
  - Reading any other file (including other files in a skill folder) takes the existing path unchanged and does not touch `active_skills` (REQ-15).
- Original `run_tool` skill hook: if `self._skill_read` was set by this call, return the skill result without ordinary tool clipping; otherwise apply the ordinary limit. Step 10 replaces that ordinary limit and its persistence behavior. Reset `self._skill_read = None` at the start of `run_tool`.
- In `ask` after `result = self.run_tool(name, args)`: if `self._skill_read` is set, replace the content that will be saved with `f"[skill loaded: {name} - {description}]"` for **both** the `history` record and `note_tool`, then reset `self._skill_read = None`. The raw body must not reach `history` or `memory["notes"]`. Multiple skills in one request all stay in `active_skills` (REQ-14).
- Note: the existing repeat guard rejects a third consecutive identical tool call; validation failures occur before that guard.

### Step 6. Lifecycle in `ask` (highest risk) [run suite]
Do this as a **pure refactor first**: wrap the existing body of `ask` after the initial `record` in `try: ... finally: self.active_skills = {}` and run the suite before adding behavior. Do not alter counters or return values.
- At the top of `ask` set `self.current_request = user_message`.
- `finally` clears `self.active_skills` on every exit, including the `RuntimeError` from `model_client.complete` (REQ-32, 33, 34).
- A skill never starts a request active unless forced (step 7).
- The body is never re-injected on later requests; `history` keeps only the `[skill loaded: ...]` notes (REQ-27).
- `main` must also leave nothing active on `/exit`, `/quit`, EOF and `KeyboardInterrupt`; since active state only exists inside `ask`, the `finally` covers it. Add a test anyway.

### Step 7. Forced `/skill` and CLI handling [run suite]
- `ask` gains an optional keyword `forced_skill=None` (a skill dict). If given, before the loop: `self.active_skills[name] = truncate_skill(body, self.skill_cap)` and add the `skills_used` entry. `user_message` stays the user's prompt only (no body), so history and `memory["task"]` hold just the prompt (REQ-17, 27). Because it is set inside the `try`, the same `finally` clears it.
- Add `parse_skill_command(self, text)` -> `(kind, value)` in `MiniAgent` or `main`:
  - `/skill` alone -> list: print each `name: description`, or `no skills found`; no model call (REQ-21).
  - `/skill <name>` with no prompt -> print `usage: /skill <name> <prompt>`; no model call (REQ-18).
  - `/skill <name> <prompt>` with unknown name -> print `unknown skill '<name>'. available: a, b` (or `none`) to stderr/stdout; no model call (REQ-19).
  - Otherwise run `agent.ask(prompt, forced_skill=skill)`.
- `main` (line 969): handle `/skill...` in the REPL loop **before** the generic `agent.ask` fall-through and **after** the built-in commands (REQ-20), and in the one-shot `args.prompt` path (REQ-21). Do not change any existing command branch. Match with `user_input == "/skill" or user_input.startswith("/skill ")` so names like `/skills` are not captured.
- Help: append `/skill` to `HELP_TEXT` (line 15) and add this row to `HELP_DETAILS` (lines 22-32), matching column style: `"/skill   Run a prompt with a skill: /skill <name> <prompt>. Bare /skill lists skills."`. These are static strings; the dynamic list comes from bare `/skill`. Check the welcome screen test still passes (it uses `HELP_TEXT` if referenced).

### Step 8. `/reset`, CLI flag, wiring
- `reset` (line 717): additionally set `self.session["skills_used"] = []`, `self.active_skills = {}`, re-run `discover_skills` (print warnings), then `self.prefix = self.build_prefix()` so the prompt reflects new/removed skills (REQ-4, 24). Keep the existing history/memory reset and save.
- `build_arg_parser` (line 945): add `--skill-max-chars` (`type=int`, `default=None`, help text, optional; REQ-30, 31). Reject values `< 1` with `parser.error` or document that `<= 0` is invalid; decide in code review and test it.
- `build_agent` (line 912): pass `model_name=args.model` and `skill_max_chars=args.skill_max_chars` to **both** `MiniAgent.from_session(...)` and `MiniAgent(...)`.
- Tests construct `MiniAgent` with no `model_name`, so they get the 24,000 tier unless a test passes `model_name` or `skill_max_chars`.

### Step 9. Docs and final run
- `README.md`: add a short "Skills" section (folder layout, required frontmatter, `/skill`, `--skill-max-chars`, the unsupported YAML features, and the known limits C2, C3, C4, C9).
- Do **not** edit `intent.md`.
- Run `python -m pytest -q` in `project_2`, confirm no new entries in `pyproject.toml` dependencies, then run the manual evaluation below.

### Step 10. Bounded result retention and retrieval [run suite]
Requirements: REQ-16, 25, 27, 29, 036, 039, 040, 044.

- [ ] T010 Add failing retention and retrieval tests listed below. Keep skill loading out of saved history and memory.
- [ ] T011 In `run_tool`, raise the ordinary-result content retention limit to 16,000 Python characters. Keep the existing separate skill-body cap and `_skill_read` handling. Record the retained length and, when larger, the original length and an explicit permanent-discard notice; keep metadata distinguishable from the retained content so offsets are unambiguous. Preserve useful command status and stderr under truncation and do not claim discarded data is retrievable.
- [ ] T012 Give stored tool events stable, session-local identifiers in `record`; backfill and persist new identifiers for pre-change events when loading a session. IDs must remain valid across save/resume and cannot refer to a different result after `/reset`. Do not persist skill bodies as results.
- [ ] T013 Add `read_tool_output` to `build_tools`, `validate_tool`, and `tool_example`, with a tool method to retrieve a bounded range by result ID and character offsets. Resolve only the current session's retained events; reject missing IDs, invalid/empty/out-of-bounds ranges, and requests for discarded content with explicit errors. Do not execute the original tool again. Define chunk size so its content **and metadata** fit within the 3,000-character prompt allowance.
- [ ] T014 Preserve normal approval checks, active-skill delegation restrictions, existing session loading, and ordinary `read_file` semantics. Retrieval is read-only and must not allow access to another session or a skill body through a saved skill-load event.

### Step 11. Recent-first transcript budgeting [run suite]
Requirements: REQ-12, 27, 28, 037, 038, 039, 044.

- [ ] T015 In `history_text`, budget the 12,000-character transcript from the newest entries backward, then render chosen entries in chronological order. Allocate at least 3,000 characters for each recent result when it fits; a just-recorded result of at most 3,000 characters, including the Lab 4 grid example, must appear in full in the next prompt. Account for headings, arguments, identifiers, and truncation markers in the budget; explicitly mark omitted content or entries.
- [ ] T016 Keep tool names and arguments identifiable, even if oversized arguments require an explicitly marked abbreviation. Preserve the existing read-after-write invalidation and avoid keeping stale reads as the only evidence of a file. Show retrieval instructions and IDs for retained content omitted from the prompt.
- [ ] T017 Verify `prompt` still includes active skills separately, the available-skills prefix remains stable, and no skill sections appear when there are no skills. Retain tests for unchanged memory and prefix behavior, but replace assertions that require the entire no-skill prompt or transcript to stay byte-identical.

### Step 12. Opt-in command-result display [run suite]
Requirements: REQ-29, 041, 044, 045.

- [ ] T018 Add an optional `--show-command-output` switch in `build_arg_parser`; pass it through `build_agent`, resumed agents, and the relevant `MiniAgent` constructor. Default off. Define child-agent behavior explicitly so delegated output is not unexpectedly duplicated.
- [ ] T019 Update `tool_run_shell` and the result presentation path to display the command, exit status, available stdout/stderr, and explicit truncation or timeout/failure information when enabled. Display only after approval and actual execution; keep it separate from the final answer. Avoid duplicate display, sanitize unsafe terminal control characters, and do not display a skill body merely because a skill was loaded.
- [ ] T020 Retain available output for failed and timed-out processes where the runtime provides it; do not invent an exit code. Keep the normal result for session/model use and maintain the retention and prompt budgets. Verify captured CLI output using `capsys`, including both enabled and default-off modes.

The display guarantee is **conditional on the option being enabled**, as specified in REQ-041; it cannot guarantee display when the terminal itself fails.

### Step 13. Bounded final-answer turn [run suite]
Requirements: REQ-27, 28, 29, 32-34, 042-044.

- [ ] T021 Refactor `_ask_loop` to issue exactly one additional, final-only model request after exhausting the tool-call budget, including when calls failed validation, were denied, or returned errors. Keep the existing maximum tool attempts and do not run any tool returned in this turn. Retain active skills in the prompt until finalization completes.
- [ ] T022 Also attempt one final-only request after exhausting malformed-response attempts or encountering a model-client `RuntimeError`, unless the user interrupts or cancels. No recursive finalization or unbounded retries. If the final request errors, is malformed/empty, or requests a tool, return an explicit failure containing the relevant reason; do not claim a report was produced.
- [ ] T023 Record the final answer or failure consistently, and retain `ask`'s `finally` cleanup. No implementation can guarantee a model-generated answer when the model service is unavailable, the user interrupts, or session persistence fails; report these limits rather than silently succeeding.
- [ ] T024 Test both skill and no-skill requests and ensure the final turn never bypasses approval or executes a model-proposed write, shell, or delegation call.

### Step 14. Documentation and final verification [run suite]
Requirements: REQ-28-30, 036-045.

- [ ] T025 Update `README.md` and `--help` for the display setting, storage and prompt limits, retrieval, final-only turn, privacy implications, and the fact that `--max-new-tokens` limits output rather than input context. Note that total session storage and subprocess memory remain unbounded.
- [ ] T026 Reconcile the outdated no-skill and step-loop guidance in `CLAUDE.md` with the revised spec before implementation; do not edit `intent.md` or unrelated plans.
- [ ] T027 Run the full agent suite and manually exercise the Lab 4 skill with an available local model, display enabled and approval set to `ask`. Check the grid is visible in the prompt/result and shown in the CLI; treat model compliance as a separate observation.

## Tests
Write each test once, in `tests/test_mini_coding_agent.py`. Use a small helper that writes `skills/<folder>/SKILL.md` under `tmp_path` from `name`, `description`, `body`. Use `capsys` for warnings (stderr).

**Discovery and parsing**
- Valid skill discovered with name, description, relative path (REQ-1).
- Loose `skills/x.md` and `skills/a/b/SKILL.md` ignored (REQ-2).
- No `skills/` or empty `skills/`: zero skills, empty stderr (REQ-3).
- Parametrized invalid skills are skipped, warning contains the file path and reason, other skill still loads: missing name; missing description; empty value; `description: |`; `description: >`; no frontmatter; unterminated frontmatter; `name: Bad_Name` (REQ-6, 6a).
- `description: "quoted"` keeps the quotes (REQ-5a).
- A line without `:` and an indented line inside frontmatter are ignored and the skill still loads; an unknown key (`license: x`) is ignored.
- Duplicate name: first folder alphabetically kept, warning for the other (REQ-7).
- Invalid UTF-8 `SKILL.md` skipped with a warning, no crash (REQ-8).
- The real `skills/train-and-report/SKILL.md` parses (smoke test using the repo path).

**Size cap**
- `skill_cap_for_model`: `qwen3.5:4b`->4000, `llama3.1:8b`->12000, `x:13b`->12000, `x:70b`->24000, `mistral`->24000, `None`->24000; override wins (REQ-31).
- `truncate_skill`: over cap ends with `[skill truncated at N chars]`; at cap unchanged (REQ-16).
- A model `read_file` of a 30,000-char body with default tier returns 24,000 chars plus the notice, and is **not** cut to 4,000 by `run_tool`.

**Prompt**
- With skills: prefix lists name, description, path, and the read-first instruction; body text absent (REQ-9, 10, 11).
- Zero skills: no available- or active-skills section appears; check unaffected prefix and memory properties without requiring the full prompt to match the old output (REQ-12, 28).
- The skills list is identical before and during an active skill (REQ-35).
- The `Active skills:` section appears in `prompt()` only while a skill is active and starts at column 0.

**Model-selected skills**
- Scripted model: `read_file` on `skills/a/SKILL.md` then `<final>`. During the second `complete` call (capture the prompt in a recording `FakeModelClient` subclass or wrapper), the prompt contains the body under `Active skills:`; `skills_used` has one entry with the request (REQ-13).
- History for that tool step contains `[skill loaded: a - <description>]` and **not** the body; same for `memory["notes"]`.
- Two skills read in one request are both in the prompt on the following step (REQ-14).
- `read_file` on `skills/a/notes.md` succeeds, does not activate anything (REQ-15).
- `read_file` of SKILL.md ignores `start`/`end` and has no line-number prefix.
- Path forms `./skills/a/SKILL.md` and `skills/a/../a/SKILL.md` also match.

**Forced skills**
- `/skill a do it` (via `main` with a patched `input`, or the helper): the model's prompt includes the body under `Active skills:` on **every** step of a multi-step request; history and `memory["task"]` contain only `do it` (REQ-17, 27).
- `/skill a` -> usage text, `FakeModelClient` call count 0 (REQ-18).
- `/skill nope x` -> error listing available skills, call count 0 (REQ-19).
- Bare `/skill` lists skills, no model call; `/help` output mentions `/skill` (REQ-21).
- One-shot `main([... "/skill", "a", "do it"])` works.
- `/skills` and other built-ins are not captured by the `/skill` branch (REQ-20).

**Lifecycle**
- Active set remains present during the final-answer turn and is empty after final answer, finalization failure, malformed-output stop, or model-client error (REQ-33).
- Active set is empty at the start of the next request and the body is absent from the next request's prompt (REQ-27, 34).
- `memory_text()` shows `skills_active` during a request (check from inside a scripted model call) and `skills_used` after it (REQ-23).
- `/exit`, EOF and `KeyboardInterrupt` leave nothing active.

**Session**
- Save then `from_session`: `skills_used` restored, `active_skills` empty (REQ-22).
- A session JSON written without `skills_used` loads, `/memory`, a skill read and `/reset` all work; a session with `"skills_used": null` also loads (REQ-25).
- Resume a session whose `skills_used` names a deleted skill: loads, stderr has the warning once (REQ-26); a malformed entry (string instead of dict) does not crash.
- `/reset` clears `skills_used` and active state, re-scans (add a new skill, reset, it appears in the prefix; delete one, it disappears) (REQ-4, 24).

**Delegation**
- A child created via the `delegate` tool lists the parent's skills in its prefix, performs no extra discovery warnings, and a skill read inside the child does not add to the parent's `skills_used`.

**Compatibility**
- Existing tests for unchanged behavior pass; update only tests that assert superseded output, transcript, or step-limit behavior (REQ-28, 29).
- No new imports of third-party packages; `pyproject.toml` dependencies unchanged; `--skill-max-chars` optional and parsed as int (REQ-30).

**Result retention and retrieval (steps 10-11)**
- Test non-skill results at 15,999, 16,000, and 16,001 characters: exact retained content/length at and below the cap; explicit discarded-length notice above it; discarded bytes cannot be retrieved (REQ-036, 040). Include multiline and non-ASCII text to verify character offsets.
- Retrieve first, middle, and last chunks by ID without rerunning the originating tool; check every returned chunk including its wrapper is visible in the next prompt. Test empty, negative, reversed, oversized, and out-of-range requests and missing, cross-session, and stale-after-reset IDs; all fail explicitly without exposing other data. Retrieving a skill-load event can return only its stored load notice, never the active body (REQ-039, 040, 044).
- Resume an older session with no result IDs, retrieve its retained contents, save/resume again, and verify ID stability and explicit handling of data already discarded by the old cap (REQ-25, 040).
- Test recent results at 2,999, 3,000, and 3,001 characters and a realistic 1,364-character result with the grid after position 1,300. Verify full immediate visibility at or below 3,000 and a clear notice plus working retrieval above it (REQ-037-039).
- Exceed 12,000 characters with old and recent history and oversized arguments; verify the transcript budget, newest-entry priority, chronological rendering, identifiable calls, and omission notices. Preserve existing duplicate-read and read-after-write regression tests (REQ-037, 038).

**Command display (step 12)**
- With display enabled, capture stdout and stderr for a successful command and a nonzero-exit command; assert command, output, exit code, and truncation notice when applicable are actually printed exactly once, separate from the final answer. Test empty stdout/stderr (REQ-041).
- Test a timeout and process-start failure with any available output and an explicit error, not a fabricated exit code. Test dangerous terminal control characters and confirm output cannot manipulate the terminal (REQ-041).
- With display disabled, confirm command output is not printed as a tool result. With approval denied, confirm the process does not run and no executed-command output is printed. Check interactive and one-shot CLI wiring, resume, and child delegation do not duplicate or unexpectedly enable output (REQ-29, 041, 044).

**Final-answer turn (step 13)**
- For both active-skill and no-skill requests, exhaust the tool limit with successful calls, invalid paths/arguments, repeated rejections, and denied or failing tools; assert exactly one extra model request sees the latest results and can produce the final answer (REQ-042, 044).
- Exhaust malformed-response attempts and simulate model-client errors; assert one bounded final attempt when possible and an explicit failure if the final call also fails (REQ-043). A user interrupt or session-persistence failure must not cause an unbounded retry.
- Return a tool request, empty output, or malformed output on the final turn; assert no requested tool runs, no extra retry occurs, and the user receives an explicit failure. Test a dangerous shell/write request to verify approvals and safety are not bypassed (REQ-042-044).
- Confirm skill bodies remain active in the final prompt but are absent from stored history and memory, then are cleared after both successful and failed finalization. Ordinary final answers before the limit must not trigger an extra model call (REQ-27, 33, 044).

**Manual evaluation (not automatable, C6)**: with a real Ollama model, run prompts such as "train the vacuum robot and report results" and confirm the model reads `train-and-report` itself, follows it, and that `/skill train-and-report <prompt>` forces it. Try a 4B model to see whether the truncation notice or the tier values need tuning (C2).

## Requirement mapping

| Requirements | Plan coverage |
|---|---|
| REQ-1-4 | Step 1 discovery and step 8 reset |
| REQ-5, 5a, 6, 6a, 7, 8 | Step 1 parsing and validation |
| REQ-9-12, 35 | Step 4 skill prompt; step 11 agent-wide prompt |
| REQ-13-16, 31 | Steps 2 and 5 skill loading and caps; step 10 preserves the cap |
| REQ-17-21 | Step 7 forced skill and CLI |
| REQ-22-27 | Steps 3, 5-8, and 10-13 session and lifecycle |
| REQ-28-30 | Steps 10-14 compatibility and verification |
| REQ-32-34 | Steps 6, 7, and 13 lifecycle |
| REQ-036 | Step 10 bounded retention (T010-T011) |
| REQ-037-038 | Step 11 transcript budget and visibility (T015-T017) |
| REQ-039-040 | Steps 10-11 retrieval, IDs, resume, and notices (T012-T016) |
| REQ-041 | Step 12 opt-in command display (T018-T020) |
| REQ-042-043 | Step 13 final-only turn and failure reporting (T021-T023) |
| REQ-044 | Steps 10-13 no-skill, permission, and body-privacy compatibility |
| REQ-045 | Step 14 documentation (T025) |

## Step 15: Similarity check, skill log and context window (REQ-046-051)
- Add `skill_tokens` and `skill_similarity` (module level, stdlib).
- `MiniAgent.match_skill` runs at the start of `ask` (inside the existing `try/finally`), logs the `match` entry and activates the selected skill through `activate_skill` with a trigger. `activate_skill` logs `executed` once per skill per request; the `finally` logs `finished` before clearing active skills.
- `MiniAgent.log_skill_event` appends to `.mini-coding-agent/skills.jsonl`; the CLI turns on the stderr echo.
- `OllamaModelClient` sends `num_ctx`; new flags `--num-ctx` and `--skill-threshold`.
- `train-and-report` description starts with "Solve Lab 4:" so that prompt passes the threshold.
- Tests: tokenizer and bounds, repo skill matches Lab 4 prompts, similarity activation + log contents, no-match logging, model and explicit triggers, single `executed` on re-read, threshold off, no log without skills, echo off by default, `num_ctx` payload.

| REQ | Covered by |
|---|---|
| REQ-046-051 | Step 15 |

## Done criteria
- Every testable behavior requirement has at least one passing test; verify documentation by inspection and selection quality manually (C6).
- Existing tests for unchanged behavior pass; changed tests match the revised no-skill, output, and final-answer requirements.
- Boundary tests verify the 16,000-character retention cap, 3,000-character recent-result allowance, and 12,000-character transcript budget; saved retained content is recoverable without rerunning a command.
- With command display enabled, executed command results and failures are visible in captured CLI output. The disabled default does not display them.
- Tool-limit and malformed-response exhaustion lead to one final-only attempt; no tool is executed during it, and a model-service failure is reported explicitly.
- Loading or forcing a skill does not persist its body in `history`, `memory`, or the saved session JSON.
- `intent.md` is unchanged.
