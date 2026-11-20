# Intent: Mini coding agent can read skills files
Authors: Jonah Harling, Caleb Vail, Minh Le

mini-coding-agent has no agent skills component. We will add one so the agent can find skill folders, pick the right skill for a prompt, and follow its instructions.
## Problem
The Mini coding agent can use models from Ollama and has session memory, but is not able to add additional context or skills. The depth of solutions given by the mini agent is therefore limited.

## Proposed Outcome
Add functionality to the mini coding agent so it can utilize any skill markdown files to solve relevant problems. It should also know when/where to use a certain skill in the development process.

## Affected Users and Systems
Users receive general solutions and cannot specialize models to their use case.
## Constraints
All current functionality of the mini coding agent is maintained.
## Open Questions
How do we define when the mini coding agent should use a specific skill?
