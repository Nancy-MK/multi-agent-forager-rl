# Multi-Agent Forager - RL Control Risk & Behavioural Unpredictability Study

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![RL](https://img.shields.io/badge/method-Q--Learning-purple) ![Multi--Agent](https://img.shields.io/badge/system-multi--agent-teal)

A controlled empirical study of independent vs. shared Q-learning policies in a multi-agent foraging environment, framed around a question that matters for agentic-AI safety: **does giving agents a shared policy make their collective behaviour safer and more predictable?**

## Why this matters for AI governance

As AI systems increasingly operate as multiple interacting agents (fleets of robots, multi-agent LLM pipelines, automated trading agents), a key open question in agentic-AI governance is how much control and predictability is gained or lost by coordinating agents versus letting them learn independently. This project runs a small, controlled experiment on that exact question, rather than a single anecdotal simulation run.

## Experimental design

- **Environment:** a grid-world foraging task where multiple agents search for and collect resources while avoiding collisions with each other
- **Two policy architectures compared:**
  - *Independent Q-tables* - each agent learns its own policy with no shared information
  - *Shared Q-table* - agents learn from and act on a common policy
- **Statistical rigour:** each configuration was run for **1,000 episodes across 8 random seeds**, so the reported effect is an average over repeated, independently-seeded runs rather than a single lucky (or unlucky) trial

## Headline finding

Policy coordination (the shared Q-table) reduced inter-agent **collisions by approximately 18%** relative to independent policies, averaged across the 8 seeds. This is treated as a quantified safety-relevant control finding: shared policies measurably reduce a specific, observable failure mode (collisions) in this environment.

## Behavioural unpredictability analysis

Beyond the headline collision metric, the project also studies how **hyperparameter choices** shape behavioural unpredictability:

- Reward shaping (individual vs. shared reward signals) and its effect on emergent cooperative or competitive behaviour
- Epsilon-annealing schedule and its effect on how quickly agent behaviour stabilises vs. continues exploring unpredictably
- Documented findings on which configurations produce more *consistent, explainable* agent behaviour, relevant to any agentic-AI system where operators need to predict what the agents will do next

## Repository structure

```
multi-agent-forager-rl/
  README.md
  requirements.txt
  src/
    environment.py     # grid-world foraging environment
    agents.py           # independent and shared Q-table agent implementations
    run_experiment.py   # runs the 1,000-episode x 8-seed comparison
    analysis.py         # collision-rate and behavioural-unpredictability analysis
```

## Getting started

```bash
git clone https://github.com/Nancy-MK/multi-agent-forager-rl.git
cd multi-agent-forager-rl
pip install -r requirements.txt

# Run the full independent-vs-shared comparison (1,000 episodes x 8 seeds)
python src/run_experiment.py --episodes 1000 --seeds 8

# Analyse the results (collision rates, reward/epsilon sensitivity)
python src/analysis.py --results-dir results/
```

## Skills demonstrated

- Multi-agent reinforcement learning (independent and shared Q-learning)
- Controlled, multi-seed experimental design for statistically meaningful comparisons
- Quantifying safety-relevant metrics (collision rate) as a control outcome
- Behavioural-unpredictability analysis relevant to agentic-AI governance

## Tech stack

Python, NumPy, Q-Learning, epsilon-greedy exploration, Matplotlib (for analysis plots)

## Licence

Developed for academic purposes. All rights reserved (c) Nancy Kamal.
