# Intent

mini-coding-agent has no agent skills component. We will add one so the agent can find skill folders, pick the right skill for a prompt, and follow its instructions.

## Done when
- A prompt like "train the vacuum robot and report the results" triggers `skills/train-and-report`.
- The agent uses that skill to solve Lab 4 (`lab4/`).
- The prompt, similarity scores and executed skill are logged.
