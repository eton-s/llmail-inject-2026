# `challenge/` — vendored subset of the original LLMail-Inject code

These files are copied **unmodified** from Microsoft's
[llmail-inject-challenge](https://github.com/microsoft/llmail-inject-challenge) repository at commit
[`3da2399`](https://github.com/microsoft/llmail-inject-challenge/commit/3da2399839b285e129888161fcc8cb03169d1b43)
(2026-09-14). The originals lived under `src/agent/`. They are © Microsoft Corporation, MIT-licensed; see
[`LICENSE`](LICENSE).

Only the pieces the replay harness actually reads or executes are kept. The rest of the original repo (the
competition web UI, the Azure Functions API, the job runner, the TaskTracker classifier weights, CI) isn't
needed to replay attacks and was removed. It remains available upstream.

| File | How `replay/` uses it |
|---|---|
| `agent/config.yaml` | tool schema, scenario definitions, sampling settings (`top_p`, `max_new_tokens`) |
| `agent/data/level1-4.json` | the benign email corpora for scenarios 1–4 |
| `agent/data/fp_tests.json` | benign emails, kept for measuring detector false positives (planned Stage 4) |
| `agent/workloads/email_retriever.py` | **executed** by `build_contexts.py`: the original RAG retriever and attacker-email filter |
| `agent/workloads/prompt_utils.py` | Spotlight constants (parsed), plus the official tool-call parser (run by the checks) |
| `agent/workloads/models.py` | imported by `prompt_utils.py` |
| `agent/workloads/llm.py` | the official prompt assembly; `tools/check_contexts.py` runs it to verify our prompts byte-for-byte |
| `agent/workloads/scenarios.py` | scoring and tool-name randomization, mirrored in `replay.py` (kept so the mirror can be audited) |
| `agent/workloads/conformal/utils.py` | `format_email`, mirrored in `replay.py` |
| `agent/workloads/detection.py` | how the 2024 detectors (Prompt Shield, TaskTracker, LLM judge) were invoked; reference for Stage 4 |

Note: the challenge actually deployed slightly different prompt strings from this `config.yaml`. The deployed
versions are the ones published with the dataset, and `build_contexts.py` uses those by default.
