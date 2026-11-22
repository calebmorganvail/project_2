# Project 2 – Agent Skills for mini-coding-agent

Adds an agent skills component to [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent) and uses it to solve Lab 4 (vacuum robot Q-learning).

## Structure
```
mini_coding_agent.py                agent + skills component (from rasbt/mini-coding-agent, extended)
tests/test_mini_coding_agent.py     agent and skills tests (fake model, no Ollama needed)
skills/train-and-report/SKILL.md    our Lab 5 skill: test, train and report the Lab 4 agent
lab4/                               Lab 4 vacuum robot: src/ (environment, q_learning, main), tests/
intent/mini_agent_reads_skills/     SDLC documents: intent.md -> spec.md -> plan.md
docs/runs/                          saved real run of the agent solving Lab 4 with the skill
CLAUDE.md                           project notes for coding agents (commands, conventions, architecture)
```

## Setup
Requires Python 3.10+ and [Ollama](https://ollama.com/download).
```bash
ollama pull qwen3.5:4b
pip install pytest numpy
```
Start the Ollama app (or run `ollama serve`) before running the agent. Use `python3` instead of `python` if that is how Python is installed on your machine.

## Run
```bash
python mini_coding_agent.py
```

To display executed shell commands and their output as well as the final answer:
```bash
python mini_coding_agent.py --show-command-output
```
Command output can contain sensitive data. Display is off by default and does not change the approval policy.

## Solving Lab 4 with the skill
1. Start Ollama and make sure the model is installed: `ollama pull qwen3.5:4b`
2. From the repository root:
   ```bash
   python mini_coding_agent.py --approval ask --max-steps 10 --show-command-output
   ```
3. At the `mini-coding-agent>` prompt, type:
   ```
   train the vacuum robot and report the results
   ```
   (or force the skill: `/skill train-and-report run lab 4`)
4. The terminal shows the similarity check and the selected skill:
   ```
   [skills] prompt='train the vacuum robot and report the results' scores[train-and-report=0.57] threshold=0.25 -> train-and-report (similarity)
   [skills] executing skill 'train-and-report' (trigger=similarity)
   ```
5. Approve the two shell commands with `y`. Following the skill, the agent runs the Lab 4 tests (`cd lab4 && python -m pytest -q`), trains the robot (`cd lab4 && python src/main.py --rows 4 --cols 4`) and reports the hyperparameters, reward curve and learned policy.
6. The prompt, similarity scores and executed skill are logged in `.mini-coding-agent/skills.jsonl`. Type `/exit` to quit.

## Results
A real run (qwen3.5:4b on a laptop) is saved in `docs/runs/`:
- `skills.jsonl`: the skill log, with three entries. `match` (prompt "train the vacuum agent and report the learned grid", score 0.596 >= 0.25, trigger `similarity`), `executed`, and `finished` (the agent's report).
- `lab4_run.txt`: the full terminal output. The Lab 4 tests passed (40 passed), training averaged 9.786 reward over the last 100 episodes, and the learned greedy policy reaches the goal (G) without entering the penalty square at (3, 1):
  ```
  > > > v
  v . v v
  > > > v
  > > > G
  ```
- `20261121-113051-08271b.json`: the session file, including `skills_used` with the trigger. Skill bodies are never stored in it.

The last exchange in `lab4_run.txt` and the session file is a shell command typed into the agent by mistake. It shows the fallback path: similarity 0.09 (below the threshold), the model selection pass chose no skill, and the command was denied.

## Changes to the original instructions
Compared with rasbt/mini-coding-agent:
- New `skills/` folder and skills component (discovery, similarity check, model selection fallback, logging).
- New command-line options: `--skill-threshold` (default 0.25), `--skill-max-chars`, `--num-ctx` (default 8192), `--show-command-output`.
- New REPL command: `/skill` (list skills) and `/skill <name> <prompt>` (force a skill).
- New log file: `.mini-coding-agent/skills.jsonl` (gitignored; copy it to keep a run).
- Larger tool-output limits, output retrieval with `read_tool_output`, and a final-answer turn after the step limit (see below).
- SDLC documents in `intent/mini_agent_reads_skills/`.

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
- For each ordinary request with available skills, the agent makes a separate model call to select matching skills from their names and descriptions, using recent user requests as context for follow-ups. It validates the response and loads the selected skill files before the first task turn. Invalid selections are retried up to three times; selection or loading failures stop the request with an error.
- The selection pass is skipped when there are no skills or when `/skill` forces one. It does not consume tool steps, but adds model inference time and compute. Selection is probabilistic: loading a skill does not prove that the task was completed.
- The model can also load a skill by calling `read_file` on its listed `SKILL.md`. If it mistakenly calls a skill name as a tool, the agent returns the correct `read_file` call without consuming a tool step; repeated mistakes remain bounded.
- Active skill bodies are included in each task prompt for the current request only, never in history, memory or the session file. The session records `skills_used`.
- **Similarity check (runs first):** every prompt is scored against each skill's name + description (cosine similarity of word counts, 0-1). If the best skill scores at least `--skill-threshold` (default 0.25) it is loaded right away and the model selection call above is skipped. Only when nothing scores high enough does the model selection pass run.
- **Skill log:** the prompt, every similarity score, the model selection result, the executed skill with its trigger (`explicit`, `similarity`, `model-select` or `model`), and the end of each skill run are appended to `.mini-coding-agent/skills.jsonl` and echoed as `[skills] ...` lines on stderr. A saved Lab 4 run is in `docs/runs/`.
- `/skill` lists skills; `/skill <name> <prompt>` forces a skill for that prompt (also works as a one-shot CLI prompt).
- Delegation is unavailable while a skill is active because child agents do not inherit active skill instructions. Delegation remains available before skill activation and after the request ends. A rejected delegation still counts toward the step limit.
- Skills do not restrict tools or enforce read-only instructions. Writing, patching and shell commands remain available subject to the configured approval policy.
- Invalid skills are skipped with a one-line `warning: skipped skills/<folder>/SKILL.md: <reason>` on stderr.
- Bodies are truncated with a `[skill truncated at N chars]` notice: 4000 chars for models <=4B, 12000 for <=13B, 24000 otherwise. Override with `--skill-max-chars N`.
- Limits: multi-line and quoted frontmatter values are not supported (quotes are kept literally; `|`/`>` values are invalid); skills added mid-session appear after `/reset`; skills must be reloaded in each new request; reading a `SKILL.md` for any reason activates it until the request ends.

## Context window
`--num-ctx` (default 8192) sets Ollama's context window in tokens. Ollama silently drops the start of a prompt that is longer than its context window, and the start is where the rules, tools and skill instructions are. The character limits above do not guarantee a token budget, so raise `--num-ctx` for long sessions if your machine has the memory.

## Limitations
- **Small model.** We used qwen3.5:4b, the agent's default, because it runs locally on a laptop. It reliably selected and followed the train-and-report skill, but it is slow (several minutes per run on CPU) and sometimes ends with a short answer such as "Done." instead of the full report. A more specific prompt helped.
- **Implementing Q-learning from scratch.** We also tried having the agent implement `QLearningAgent` from an empty class. The 4B model could not do it reliably: it sometimes answered without calling any tool, called `patch_file` with missing arguments, and its first draft had bugs in 5 of 7 methods (a list instead of a dict, undefined variables, overwritten inputs). The Lab 4 tests caught every bug. A larger model would likely do better (`--model`).
- **Word-overlap similarity.** The similarity check counts shared words, so it misses synonyms ("clean the floor" does not match "vacuum"); the model selection pass covers those cases. Requests that share words with a skill's description can trigger it. For example, "implement the QLearningAgent class in lab4/src/q_learning.py" scores 0.43 against train-and-report, a read-only skill. `--skill-threshold 1.1` turns automatic selection off.

## Test
```bash
python -m pytest            # agent
cd lab4 && python -m pytest # Lab 4
```
The agent tests use fake model responses; neither pytest suite requires Ollama. For real-model skill-selection testing, Ollama must be running with `qwen3.5:4b` installed (`ollama pull qwen3.5:4b`).

## Credit
`mini_coding_agent.py`, `tests/test_mini_coding_agent.py` and `LICENSE` (Apache 2.0) come from rasbt/mini-coding-agent.
