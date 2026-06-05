"""Decoupled omni deployment: thinker process + talker process via file IPC.

This is the minimal viable "decoupled" topology for Qwen2.5-Omni on RTP:
each stage runs in its own Python subprocess (separate process tree, separate
GPU context, can be on separate machines if the IPC file is on shared FS).

Communication is via a JSON+base64-encoded file payload containing the
thinker's text output, token IDs, and per-token last-layer hidden states.
The talker process reads this and conditions its codec generation on the
received hidden states.

This is simpler than a proper gRPC streaming topology (which would require
extending GenerateOutputPB with a hidden_states field and a C++ rebuild)
and proves the topology end-to-end. The gRPC variant is left for a
follow-up; see docs/omni-validation-report.md §3.
"""
