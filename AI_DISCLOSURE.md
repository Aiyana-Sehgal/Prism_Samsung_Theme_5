# AI Disclosure

**Team Knox — Samsung PRISM GenAI Hackathon, Theme 05**

We used AI coding assistants during this project and are documenting that openly.

## What is the team's own work

The **architecture, research and planning** are ours:

- The core idea: that the failure mode in interruptible voice agents is *timing*, not comprehension
  — the model usually picks the right tool, but acts before the user has finished changing their
  mind.
- The **commit gate** design: that a proposed tool call should be held until the user has been
  silent for a quiet window, and dropped entirely if they resume speaking.
- The decision to gate **every** tool, including read-only ones, after analysing how FDB-v3's
  strict pass rate treats any extra call as a scenario failure.
- The **idempotency** model, and the insight that the correct policy differs by domain —
  session-level deduplication for the benchmark's stateless mock tools, but **state-based**
  idempotency for the smart-home extension, where a genuine repeat later must still work.
- The decision to keep the session alive past the harness disconnect so gated calls can complete.
- The choice of benchmark, the evaluation approach, the failure analysis, and the interpretation of
  all results reported in the README.
- Direction and review of every change made during implementation.

## What AI assistants did

The **implementation**. AI coding assistants were used throughout to write, port and debug code
from our designs, including:

- the initial implementation of the agent, the controller and the smart-home extension;
- porting the system from hosted OpenAI APIs to a fully local model stack
  (`agent/local_models.py`, and the corresponding changes to `reproduce.sh`) so that reproduction
  requires no paid API key;
- debugging reproducibility defects found while validating `reproduce.sh` on a clean machine;
- drafting documentation.

## Summary

The architecture and the reasoning behind it are the team's. The code that realises them was
written with AI assistance. Every design decision, the evaluation methodology, and the conclusions
drawn from the results are our own.
