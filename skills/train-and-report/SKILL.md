---
name: train-and-report
description: Solve Lab 4: train the vacuum robot Q-learning agent in lab4, run its tests, and report the hyperparameters, learned policy and results.
---

# Train and Report

Run the Lab 4 vacuum robot end to end and report the results. Do not edit any files.

## Steps
1. Confirm `lab4/src/environment.py`, `lab4/src/q_learning.py` and `lab4/src/main.py` exist. If one is missing, stop and say which.
2. Run the tests: `cd lab4 && python -m pytest -q`. If any fail, stop and report the failing tests.
3. Train: `cd lab4 && python src/main.py --rows 4 --cols 4`
4. Read the hyperparameter constants at the top of `lab4/src/main.py`.

## Report
- Hyperparameters: living reward, discount factor, learning rate, epsilon (start, min, decay), episodes
- Learned policy: the arrow grid printed by `main.py`
- Tests passed / failed
- Average reward over the last 100 episodes
- Whether the greedy path reaches the goal and avoids the penalty square
