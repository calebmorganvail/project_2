# Spec

## Skill format
`skills/<name>/SKILL.md`:
```markdown
---
name: <name>
description: <one line, used for matching>
---
<instructions>
```

## Component (`agent_skills.py`)
- **Discover:** load every `skills/*/SKILL.md`; skip files missing `name` or `description`.
- **Match:** score the prompt against each skill's name + description (cosine similarity of word counts, 0–1). Pick the best skill if it scores ≥ 0.25. `/<name>` in the prompt triggers that skill directly.
- **Use during execution:** a `use_skill(name)` tool returns a skill's instructions so the agent can load one mid-task.
- **Execute:** add the selected skill's instructions to the prompt. The agent carries them out with its existing tools and approval rules.
- **Log:** append JSON lines to `.mini-coding-agent/skills.jsonl`:
```json
{"time": "...", "event": "match", "prompt": "...", "scores": {"train-and-report": 0.61}, "selected": "train-and-report", "trigger": "similarity"}
{"time": "...", "event": "executed", "skill": "train-and-report", "trigger": "similarity"}
```
`trigger` is `similarity`, `explicit` or `tool`.

## Tests
Using `FakeModelClient`: discovery, matching above/below threshold, explicit trigger, `use_skill`, and log output.
