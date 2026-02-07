# Learning to Play Settlers of Catan with Multi-Agent Reinforcement Learning

This project explores learning and strategic behavior in *Settlers of Catan* using reinforcement learning and multi-agent methods. We build a custom Catan simulator designed for experimentation with learning agents, planning agents, and game-theoretic analysis.

The goal is to study how agents learn placement, trading, and development strategies over repeated games, and how different learning paradigms interact in a multi-agent environment.

A full write-up for this project will be available here: TBD

---

## Project Overview

This repository includes:

* A custom Catan environment written in Python
* Baseline heuristic agents
* Reinforcement learning agents (algorithm TBD)
* Planning and forward-search agents built on top of learned policies
* Evaluation and analysis tools for multi-agent behavior

The project is intended for research and experimentation rather than strict rule-level fidelity.

---

## Dependencies

Main dependencies:

* python >= 3.x
* numpy == TBD
* torch == TBD
* gym or gymnasium == TBD
* pygame == TBD

Install locally with:

```
pip install -e .
```

---

## Running an Interactive Game

To launch an interactive game where all players are controlled manually:

```
python main.py
```

This starts a local Catan game instance with human-controlled players.

---

## Playing Against Trained Agents

Once trained agents are available, you can play against them by specifying policies on the command line:

```
python play.py \
  --policy1 "human" \
  --policy2 "RL_TBD" \
  --policy3 "RL_TBD" \
  --policy4 "RL_TBD"
```

Here, `RL_TBD` refers to a saved model checkpoint (naming convention to be finalized).

Trained agents will be stored in a directory such as:

```
RL/results/
```

---

## Forward Search / Planning Agents (Optional)

The project may include planning-based agents that use a learned policy for evaluation or rollouts.

Example usage (placeholder):

```
python play.py \
  --policy1 "human" \
  --policy2 "forward_search_TBD" \
  --policy3 "forward_search_TBD" \
  --policy4 "forward_search_TBD"
```

These agents may support additional configuration options such as:

* number of CPU processes
* thinking time per decision

Exact behavior and flags are TBD.

---

## Training Agents

To train a reinforcement learning agent from scratch:

```
python RL/train.py
```

Training details:

* Algorithm: TBD
* Hardware requirements: TBD
* Expected training time: TBD

Experiment configuration options will live in:

```
RL/config/
```

or

```
RL/arguments.py
```

(final structure TBD)

---

## Repository Structure (Tentative)

```
multiagent-catan/
├── env/
├── agents/
├── RL/
├── analysis/
├── ui/
├── main.py
├── setup.py
├── README.md
└── ...
```

---

## Status

This project is under active development. APIs, file structure, and training procedures are subject to change.

---

## Notes

This project is for educational and research purposes and is not affiliated with the official Settlers of Catan game or its publishers.