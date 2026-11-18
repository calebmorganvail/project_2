# Project 2 – Agent Skills for mini-coding-agent

Adds an agent skills component to [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent) and uses it to solve Lab 4 (vacuum robot Q-learning).

## Structure
```
mini_coding_agent.py   agent (from rasbt/mini-coding-agent)
agent_skills.py        skills component (in progress)
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
Skill decisions are logged to `.mini-coding-agent/skills.jsonl`.

## Test
```bash
python -m pytest            # agent
cd lab4 && python -m pytest # Lab 4
```

## Credit
`mini_coding_agent.py`, `tests/test_mini_coding_agent.py` and `LICENSE` (Apache 2.0) come from rasbt/mini-coding-agent.
