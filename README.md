# Pacman Multi-Agent AI

A two-agent decision system for **Pacman Capture the Flag**, built with adaptive strategy switching and feature-based Q-value action selection.

| Course tournament | Decision time during development |
|---|---|
| **802 EP** | **~0.8 s -> ~0.3 s per action** |

## Overview

This project explores decision-making in a competitive, partially observable environment where two agents must collect food, defend their territory, avoid opponents, and return carried food safely.

The final runtime agent uses rule-based mode switching to choose between **attack**, **defence**, and **go-home** behaviours. Within the selected mode, it evaluates every legal action using a feature-based Q-value function and executes the highest-valued action.

PDDL was used during development to formalise strategic goals, actions, and constraints. Online PDDL planning was disabled in the final decision loop to improve timing stability; the resulting runtime policy uses strategic rules informed by that modelling work.

## Key Features

- **Adaptive mode switching:** responds to role, visible invaders, nearby defenders, carried food, score, remaining time, and map scale.
- **Q-value action selection:** scores legal actions with mode-specific feature and weight sets for offence, defence, and safe return.
- **Feature and reward engineering:** balances food collection, scoring progress, opponent risk, defensive interception, return safety, and movement efficiency.
- **Multi-agent coordination:** assigns initial attacker and defender roles and incorporates teammate position into defensive action evaluation.
- **Map generalisation:** normalises distance and territorial-depth features by board scale so the same weights transfer more consistently across layouts.
- **Runtime optimisation:** caches repeated opponent-state calculations and reduces expensive distance work. The project report records an improvement from approximately **0.8 seconds to 0.3 seconds per action** during development.

## Architecture

```text
Observed game state
        |
        v
State and risk analysis
(role, food, score, time, opponents, map scale)
        |
        v
Adaptive mode selection
  attack | defence | go_home
        |
        v
Mode-specific feature extraction
        |
        v
Q(s, a) = weights . features(s, a)
        |
        v
Highest-valued legal action
```

The competition build deploys fixed weights: online training and exploration are disabled at runtime. Reward functions and weight-update code remain in the project to document the development and tuning workflow.

## My Contributions

My work focused on the agent layer rather than the underlying Pacman engine:

- Designed the adaptive attack, defence, and return-home decision rules.
- Implemented mode-specific Q-value action selection.
- Engineered offensive, defensive, escape, risk, and teammate-aware features.
- Designed shaped rewards for food collection, successful deposit, defensive outcomes, and risk management.
- Normalised map-dependent features to improve transfer across different layouts.
- Tuned and deployed fixed Q-value weights after development and training.
- Added role specialisation and basic teammate-aware coordination.
- Reduced repeated computation through caching and simplified feature calculations.
- Used PDDL models during development to reason about high-level goals and constraints, then removed online PDDL planning from the final action loop.

## Results

- Achieved **802 EP** in the course tournament evaluation.
- Tested during development across multiple official layouts and against multiple opponent teams.
- Reduced reported average decision time from approximately **0.8 s to 0.3 s per action** through caching and feature-computation changes.
- Improved cross-map behaviour by scaling distance-based features to the dimensions of each layout.

These figures are results reported during project development and course evaluation; they are not presented as a general benchmark outside that environment.

## Repository Structure

```text
pacman-multi-agent-ai/
├── README.md
├── docs/
│   └── Project_Report.pdf
└── src/
    ├── myTeam.py              # Agent logic, features, rewards, and weights
    ├── myTeam.pddl            # PDDL model retained from the design workflow
    ├── QLWeightsMyTeam.txt    # Deployed feature weights
    ├── capture.py             # Capture the Flag simulator entry point
    ├── layouts/               # Game layouts
    └── ...                    # Framework and support files
```

## Running the Project

This repository is based on the course-provided Python 3 Capture the Flag environment. The supplied runtime must include the framework files under `src/` and the course's `lib_piglet` package, which `myTeam.py` imports. No standalone installation command is provided here because that package was distributed through the course environment.

From the repository root:

```bash
cd src
python capture.py -r myTeam -b berkeleyTeam
```

To inspect the simulator's supported layouts, display modes, team options, and multi-game settings:

```bash
python capture.py --help
```

`myTeam` can also be assigned to the blue side with `-b`. Team arguments refer to the corresponding Python team modules in `src/`.

## Technical Report

For the design process, feature definitions, reward shaping, evaluation discussion, and critical reflection, see the [Technical Project Report](docs/Project_Report.pdf).

## Limitations / Future Work

- **Opponent modelling:** classify opponent styles and adapt risk thresholds or weights accordingly.
- **Stronger coordination:** replace fixed roles with dynamic role reassignment and explicit target allocation.
- **Planning depth:** add short-horizon look-ahead to reduce greedy choices near traps and dead ends.
- **Context-dependent policies:** vary weights based on score, remaining time, and game phase.
- **Joint learning:** explore shared rewards or multi-agent reinforcement learning instead of independently evaluated actions.

## Attribution

This is an academic project developed at **Monash University** using a Python 3 adaptation of the **UC Berkeley Pacman Capture the Flag framework**.

The game engine, graphics, capture-agent APIs, utilities, layouts, baseline opponents, and initial course scaffold were provided by UC Berkeley and/or Monash University. I do not claim authorship of those components. My contributions are the strategy switching, feature and reward design, Q-value policy configuration, coordination logic, runtime optimisation, testing, and analysis described above.

Original framework reference: [UC Berkeley Pacman Capture the Flag](https://ai.berkeley.edu/contest.html)
