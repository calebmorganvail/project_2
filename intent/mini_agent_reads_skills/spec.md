# Spec: Mini coding agent reads skills
Source: [intent.md](./intent.md)

## Scope update: agent-wide output and reporting
The output and final-answer requirements below apply to every request, whether or not a skill is available or active. They intentionally change the original no-skill baseline while preserving skill discovery, activation, permissions, and unrelated existing behavior. The original intent remains unchanged.

## Decisions (confirmed with the team)
| Topic | Decision |
|---|---|
| Discovery | `skills/` in the workspace root only |
| Layout | `skills/<folder>/SKILL.md` only |
| Metadata | Every skill must have a `name` and a `description` (YAML frontmatter) |
| Selection | Three triggers: (1) **similarity**: each prompt is scored against every skill's name + description and the best skill at or above the threshold (0.25) is loaded automatically (REQ-046-047); (2) **model**: otherwise the model may load a skill itself with `read_file`; (3) **explicit**: `/skill <name> <prompt>` (the `/skill` prefix avoids collisions with built-in commands). Added to meet the assignment requirement to log the prompt, the similarity check and the executed skill. |
| Loading | Lazy: only name + description in the prompt; the model loads the body with the existing `read_file` tool on the skill path |
| Multiple skills | Allowed, no cap on the number of skills |
| Invalid skills | Skipped with a warning; agent keeps running |
| Active state | A skill is active only for the current request, including the final-answer turn after the tool limit. It becomes inactive when the request finishes, errors, or exits. Usage is recorded in the session |
| Extra files | The agent may read other files in a skill folder via `read_file` |
| Size cap | Per-skill body cap in characters, tiered by model parameter count parsed from the model name: <=4B: 4,000; <=13B: 12,000; larger or unparsable: 24,000. Overridable with `--skill-max-chars`. Over-cap bodies are truncated with a visible notice |
| Frontmatter parser | A minimal built-in `key: value` parser; no PyYAML dependency. **Multi-line values and quoted values are not supported** (design decision) |

## Requirements
Each requirement is testable. Tests go in `tests/test_mini_coding_agent.py` using `FakeModelClient`.

### Discovery
- **REQ-1** On agent start, every `skills/<folder>/SKILL.md` under the workspace root is discovered.
- **REQ-2** Only direct subfolders of `skills/` are scanned; loose `.md` files in `skills/` and nested deeper folders are ignored.
- **REQ-3** If `skills/` does not exist or is empty, the agent starts normally with zero skills and prints no warning.
- **REQ-4** Skills are discovered at startup and re-scanned on `/reset`. (See concern C4.)

### Metadata parsing
- **REQ-5** `name` and `description` are read from the frontmatter block (between `---` lines) at the top of `SKILL.md` by a built-in single-line `key: value` parser. No third-party YAML library is used.
- **REQ-5a** Multi-line values (block scalars such as `|` or `>`, or continuation lines) and quoted values are not supported. Quote characters are kept literally as part of the value, and a `|` or `>` value is treated as invalid (see REQ-6).
- **REQ-6** A skill missing `name`, missing `description`, with an empty value for either, with a block-scalar value, or with a missing/unterminated frontmatter block is skipped and a warning naming the file and reason is printed. Other skills still load.
- **REQ-6a** A skill `name` must match `[a-z0-9-]+`; otherwise the skill is skipped with a warning.
- **REQ-7** Two skills with the same `name`: the first in alphabetical folder order is kept, the other is skipped with a warning.
- **REQ-8** An unreadable file (permissions, invalid UTF-8) is skipped with a warning and does not crash the agent.

### Prompt construction (lazy loading)
- **REQ-9** When at least one skill exists, the prompt prefix includes an "Available skills" section listing each skill's name, description, and `SKILL.md` path (workspace-relative).
- **REQ-10** The prefix contains instructions telling the model to `read_file` a skill's `SKILL.md` when the task matches its description, and to follow it.
- **REQ-11** The prefix does not contain the body of any skill that has not been loaded.
- **REQ-12** With zero skills, the prompt has no available-skills or active-skills section. Agent-wide prompt and output changes in REQ-036 through REQ-045 also apply when there are no skills.

### Model-selected skills
- **REQ-13** When the model calls `read_file` on a discovered skill's `SKILL.md` path, the skill becomes active for the current request and is recorded as used in the session.
- **REQ-14** The model may load multiple skills in one request; all are active simultaneously, with no cap on the number.
- **REQ-15** The model may `read_file` other files inside a skill's folder; these succeed under the existing workspace-path rules and do not change the active set.
- **REQ-16** Skill body size cap: when a skill's `SKILL.md` body is delivered to the model (via `read_file` or forced injection), it is truncated to the cap for the current model (see REQ-31) and ends with a visible notice, e.g. `[skill truncated at N chars]`. Bodies at or under the cap are unchanged.
- **REQ-31** The cap is chosen from the parameter count parsed from the `--model` name (e.g. `qwen3.5:4b` -> 4B): <=4B -> 4,000 chars; >4B and <=13B -> 12,000; >13B or no parsable size -> 24,000. `--skill-max-chars N` overrides the tier value.

### User-forced skills
- **REQ-17** `/skill <name> <prompt>` runs `<prompt>` with the named skill loaded: the skill's `SKILL.md` content (subject to REQ-16) is injected into the prompt for that request, and the skill is active for that request.
- **REQ-18** `/skill <name>` with no prompt prints a usage message and does not call the model.
- **REQ-19** `/skill <name> ...` where `<name>` is not a discovered skill prints an error listing available skills and does not call the model.
- **REQ-20** Built-in commands are unchanged. Because skills are only invoked through `/skill`, skill names cannot collide with them and no collision handling is needed.
- **REQ-21** `/help` documents `/skill <name> <prompt>`; `/skill` with no arguments lists discovered skills. One-shot CLI prompts also accept `/skill <name> <prompt>`.

### Active state lifecycle
- **REQ-32** A skill is active only while the model is using it to work on the current request.
- **REQ-33** All skills become inactive when the request ends for any reason, including a final answer, failure of the final-answer turn, an error, or user exit (`/exit`, `/quit`, EOF/interrupt). Reaching the tool limit alone does not deactivate skills before the final-answer turn.
- **REQ-34** Skills are never active at the start of a request unless forced by `/skill` or selected by the similarity check (REQ-047) for that request.
- **REQ-35** The "available skills" list in the prompt prefix (REQ-9) is not affected by active state.

### Session and memory
- **REQ-22** The session JSON records which skills were used (a `skills_used` list of names, with the request they were used in). Active state itself is transient and is not restored by `--resume`.
- **REQ-23** `/memory` shows skills used in the session; during a request, currently active skills are shown as active.
- **REQ-24** `/reset` clears skill usage records along with history and memory.
- **REQ-25** Sessions saved before this feature (no `skills_used` field) load without error.
- **REQ-26** A resumed session that references a skill no longer on disk keeps loading; a warning is printed.
- **REQ-27** After a request ends, the skill body is not re-injected on later requests; the model must load it again if needed. History retains the names of skills used.

### Backward compatibility
- **REQ-28** Existing tests for unchanged behavior continue to pass. Tests asserting the superseded no-skill prompt, output limits, or step-limit behavior are updated to match the agent-wide requirements.
- **REQ-29** Existing tool permissions, approval flow, normal tool-call parsing, `--resume`, and existing slash commands remain compatible with and without skills, except for the explicitly specified agent-wide output and final-answer changes.
- **REQ-30** No new required CLI arguments or third-party dependencies are introduced. `--skill-max-chars` is optional.

### Agent-wide output and final reporting
- **REQ-036** Each non-skill tool result retains up to 16,000 characters in the session, up from 4,000. If the result exceeds that limit, the retained result explicitly states that the remainder was discarded and cannot be retrieved. This limit does not replace the separate skill-body cap in REQ-16.
- **REQ-037** Recent tool results in the model's transcript have an allowance of at least 3,000 characters per result, subject to the 12,000-character transcript budget. When the budget is exceeded, the transcript prioritizes newer entries and presents selected entries in chronological order, with tool names and arguments identifiable.
- **REQ-038** Any result or transcript entry shortened for the model explicitly indicates that content was omitted. A result retained in full up to 3,000 characters, including the 1,364-character Lab 4 output with the policy grid near its end, is visible in the immediately following model prompt.
- **REQ-039** The model can retrieve a bounded range of a retained prior tool result from the current session without rerunning the original tool. The result identifier and retrieval instructions are available to the model when retained content is omitted from its prompt; each retrieved range fits within the recent-result allowance.
- **REQ-040** Retrieval reports invalid identifiers, invalid ranges, and unavailable or discarded content as errors. Sessions saved before retrieval identifiers were introduced remain loadable, and their retained tool results can be identified and retrieved.
- **REQ-041** Command-result display is an optional user-controlled CLI setting, disabled by default. When enabled, each executed shell command displays its command, available stdout and stderr, and any truncation notice, separately from the model's final answer. It displays the exit status when available, or an explicit error when the command times out or cannot complete. Denied commands are not represented as executed.
- **REQ-042** Once the tool-call limit is reached, the agent makes one additional model request for a final answer or an honest partial-result report. The final-answer turn does not grant another tool call; any tool call returned in that turn is not executed.
- **REQ-043** If the final-answer turn does not produce a usable answer, the agent returns an explicit failure message rather than claiming success. The existing limit on malformed model responses remains bounded.
- **REQ-044** Output retention, retrieval, display, and the final-answer turn apply equally to requests with and without skills. They do not bypass approval, persist active skill bodies in session history or memory, or change the skill-body size cap.
- **REQ-045** Documentation describes output retention and truncation, retrieval, the opt-in display setting, the final-answer turn, and the distinction between answer-length and input-context limits.

### Similarity check and skill log
Added because the assignment requires the component to log the prompt, the similarity check and the executed skill.
- **REQ-046** Similarity between a prompt and a skill is the cosine similarity (0-1) of their word-count vectors over the skill's name and description. Words are lower-cased, stopwords removed, lightly stemmed, and letters and digits split (`lab4` -> `lab 4`). Stdlib only.
- **REQ-047** At the start of every top-level request with at least one skill, the prompt is scored against every skill. If the best score is >= the threshold (default 0.25, `--skill-threshold`), that skill is activated for the request exactly like a forced skill (trigger `similarity`). Below the threshold nothing is activated and the model may still load a skill with `read_file` (trigger `model`). A `/skill` forced skill always wins (trigger `explicit`). Delegated child agents do not run the check.
- **REQ-048** Every decision is appended as one JSON line to `.mini-coding-agent/skills.jsonl`: a `match` entry for every request (prompt, all scores, threshold, selected skill or null, trigger), an `executed` entry when a skill becomes active (skill, trigger, path, characters given to the model; once per skill per request), and a `finished` entry per active skill when the request ends (final answer clipped to 500 characters). Each entry has a timestamp and the session id. `skills_used` entries also record the trigger. In the CLI each entry is echoed as a one-line `[skills] ...` message on stderr; library use is silent by default.
- **REQ-049** The skill log never contains skill bodies.
- **REQ-050** With zero skills nothing is scored and no log file is written.
- **REQ-051** The Ollama request sets `num_ctx` (default 8192, `--num-ctx`) so long prompts are not silently cut at Ollama's small default context window.

## Flagged concerns
- **C1 Name collisions (resolved).** Skills are invoked via `/skill <name>`, so they cannot collide with built-ins. Names are still restricted to `[a-z0-9-]` (REQ-6a); confirm this is wanted.
- **C2 Size cap tiers.** Truncation can cut a skill's later steps, changing its behavior for small models (the visible notice mitigates this). The tier values (4,000 / 12,000 / 24,000 chars) are initial guesses to tune, and parsing size from the model name is unreliable for names like `llama3` or `mistral` (they fall in the largest tier). The cap applies to `SKILL.md` bodies only; other files are subject to normal read ranges and tool-result limits.
- **C3 Context loss across requests.** Skills must be loaded again on later requests (REQ-27). Active skill bodies are separate from the transcript budget, but a long combined prompt can still exceed the model's context capacity; character limits do not guarantee a token budget.
- **C4 Hot reload.** Skills added during a session are only seen after `/reset` or restart. Confirm acceptable.
- **C5 Frontmatter parser limits (decided).** A minimal parser is used instead of PyYAML. Multi-line and quoted values are unsupported by design (REQ-5a), so skills written for other tools (which often use quoted or folded descriptions) may be skipped or have literal quotes in their description.
- **C6 Reliance on a small model.** When no skill passes the similarity threshold, whether the model picks the right skill via the prompt list is not deterministically testable; tests only verify the prompt contents and that loading is recorded. Selection quality needs manual evaluation. The similarity check (REQ-046-047) is deterministic and tested.
- **C7 Selection criterion is the description.** This resolves the intent's open question ("when to use a skill"): the description is scored against the prompt by the similarity check (REQ-047), and below the threshold the model matches the task to each description itself. Skill authors must write good descriptions.
- **C8 Prompt injection / trust.** Skills come from the workspace, so a cloned repo's skill can steer the agent (e.g. toward `run_shell`). Existing approval prompts for risky tools still apply (REQ-29); no extra trust gate is specified.
- **C9 Active-state detection.** Reading a `SKILL.md` for any reason (e.g. the user asks to summarize it) marks it active until the request ends. Also, "active" ends only at request end, not mid-request after the model last uses the skill; the agent cannot reliably tell when the model stopped relying on it.
- **C10 Intent-vs-decision change.** The intent described "any skill markdown files"; the final decision narrowed this to `skills/<folder>/SKILL.md` with required frontmatter. Intent text may need updating.
- **C11 Output retention and privacy.** The per-result cap does not bound total session size or subprocess capture memory. Command output can contain sensitive data; opt-in display and retained session data should be handled accordingly.
- **C13 Similarity vs. model choice.** Word-overlap similarity is simple and explainable but misses synonyms; a vague prompt may score below the threshold (the model can still load the skill) and a prompt sharing words with a description may load it unnecessarily. Tune the threshold and descriptions; `--skill-threshold 1.1` effectively turns automatic selection off.
- **C12 Reporting reliability.** A final-answer turn and visible results improve the opportunity to report, but they do not guarantee that a model follows the requested report format or avoids repetition.
