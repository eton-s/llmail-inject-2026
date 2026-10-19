# LLMail-Inject 2026: do year-old prompt injections still work?

A defensive-security research project (Humber College, CCGC5003 Project 1) that replays attacks from Microsoft's
[LLMail-Inject](https://github.com/microsoft/llmail-inject-challenge) prompt-injection challenge (Dec 2024 – Feb 2025)
against newer language models. The question: **of the attacks that fully succeeded against the 2024 challenge
models, how many still work on 2026 models, and how should that be measured fairly?**

## Background

In LLMail-Inject, an LLM email assistant answers a user's question about their inbox, and an attacker plants one
email that tries to make the assistant call its `send_email` tool. This project focuses on **Scenario 4**, the
hardest phase-1 setting. The attacker's email must be retrieved by RAG, slip past a defense (Prompt Shield,
TaskTracker, LLM judge or Spotlight), and get the model to pull a secret figure (`$9.378 million`) from *another*
email and send it to the attacker.

## Approach

- **Faithful prompts.** Every per-attack prompt is rebuilt with the challenge's own code (its retriever runs
  unmodified) and verified **byte-identical** to the original prompt assembly for all 2,538 Scenario 1 and 4
  prompts. Phase 1 actually deployed the prompt strings published with the dataset, not the repo's `config.yaml`.
  The dataset's version reproduces the 2024 retrieval labels for 1,003 of 1,008 attacks; the repo's, for 945.
- **The original model as baseline.** Phi-3-medium-128k-instruct, bf16, pinned to the revision live during the
  challenge, served with vLLM on Modal. It runs one request at a time with chunked prefill and prefix caching off,
  because vLLM selects Phi-3's long-context RoPE scaling per *batch*.
- **A model ladder.** Gemma-2-27B (2024), Phi-4 (Dec 2024), Gemma 4 31B (2026) and Claude Haiku 5.5 (2026) via
  OpenRouter. Every model gets the identical JSON-in-prompt tool format and the original sampling settings
  (`top_p` 0.92, temperature 1.0), with 3 draws per attack.
- **Two scores.** *Strict* is the challenge's exact-match rule, using its own parser. *Relaxed* asks whether the
  secret reached the attacker's address in any wording or format. Relaxed is the intent signal.

## Results so far: Scenario 4

212 attacks that fully exfiltrated the secret in 2024, 3 draws each (per-draw rate pooled over draws, 95% CI):

| Model | Released | Strict | Relaxed |
|---|---|---|---|
| Phi-3-medium (the original model) | 2024-05 | 6.8% [5–9] | 23.0% [20–26] |
| Gemma-2-27B | 2024-06 | 5.8% [4–8] | 19.3% [16–23] |
| Phi-4 | 2024-12 | 0.8% [0–2] | 17.9% [15–21] |
| Gemma 4 31B | 2026-04 | 9.4% [7–12] | 9.9% [8–12] |
| Claude Haiku 5.5 | 2026-10 | 0.0% [0–1] | 0.0% [0–1] |

![Headline results](replay/figures/fig1_headline.png)

**Key findings**

1. **S4 attacks transfer to a current model, but not to all of them.** Gemma 4 leaks the secret at or above the
   original Phi-3's own replay rate under strict scoring; Claude Haiku 5.5 resisted all 577 attempts.
2. **Gemma 4's leaks are reproducible.** Of attacks that leak on at least one draw, 83% leak on *every* draw,
   versus 20–31% for the 2024-era models. A stable set of about 20 attacks reliably defeats it.
3. **The metric decides whether the newer Gemma looks safer.** Compared with Gemma-2, Gemma 4 attempts to leak
   significantly less often (21% → 13%, p = 0.02). But it executes almost every attempt perfectly (21/22 vs 11/35),
   so its exact-match score is higher (7% → 13%, not significant).
4. **Phi-4's near-zero strict score is a formatting artifact.** It tries to leak about as often as its peers but
   rarely produces the required one-line JSON.
5. **Results must be read against the original model, not 100%.** The 2024 successes were mostly rare wins: in
   the raw data, identical resubmissions of a successful attack re-succeeded only 3–10% of the time. The replay
   still preserves attack strength: on the same Phi-3, 2024 successes replay far above 2024 failures (strict 7%
   vs 1%, p = 0.005; relaxed 34% vs 11%, p = 2e-6). An earlier Scenario 1 figure of 60% turned out to be a
   greedy-decoding artifact; sampled faithfully, it is 8%.
6. **Spotlight (datamarking) still blocks every model**: zero leaks on the Spotlight levels.

![Reproducibility of leaks](replay/figures/fig4_consistency.png)

The full write-up with caveats is in [`replay/FINDINGS_s4.md`](replay/FINDINGS_s4.md); the analysis notebook is
[`replay/s4_analysis.ipynb`](replay/s4_analysis.ipynb).

## Repository layout

| Path | Contents |
|---|---|
| `replay/build_contexts.py` | builds the exact per-attack prompts and checks retrieval against the 2024 labels |
| `replay/replay.py` | runs a model over the prompts; strict and relaxed scoring; resumable multi-draw runs |
| `replay/tools/` | fidelity checks (`check_contexts.py`, `test_scoring.py`), analysis and figures (`s4_*.py`) |
| `replay/serve_phi3/` | the Phi-3 baseline server on Modal, with a runbook |
| `replay/results/` | raw results (JSONL), one file per model and attack set |
| `replay/figures/` | talk figures, regenerated by `tools/s4_figures.py` |
| `challenge/` | an unmodified subset of the original challenge code that the replay depends on ([details](challenge/README.md)) |

## Running it

Python 3.11. Install with `pip install -r requirements.txt`, then work from `replay/`:

```
python build_contexts.py --scenario 4         # build prompts (once per scenario)
python tools/check_contexts.py                # verify them against the original code
python replay.py --scenario 4 --model openai --model-id google/gemma-4-31b-it --reasoning none --draws 3
python tools/s4_figures.py                    # regenerate figures
```

Two inputs live outside this repository, and their paths are currently hardcoded:
- the [LLMail-Inject dataset](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) (`DATASET` in
  `build_contexts.py`);
- the attack subset produced by a separate exploratory notebook (`DEFAULT_INPUT` in `replay.py`).

The attack texts themselves are not redistributed here.

## Next steps

- **Stage 2, "smarter = riskier?":** replay the 457 attacks that reached the 2024 model but didn't exfiltrate,
  separating willingness to leak from execution, plus a reasoning-on ablation.
- **Stage 3, "behind the old defenses?":** combine the 2024 detector verdicts with new-model compliance.
- **Stage 4, modern detectors:** score the same attacks with current guard models and compare with the 2024 detectors.
- **Robustness:** more models (Qwen3.6-27B, Ministral 3 14B), native tool-calling, temperature sensitivity (the
  2024 server's default is unknown), and a Scenario 1 re-run of the modern models on the corrected prompts.
- **Portability:** replace the hardcoded input paths with configuration.

## Attribution

This project builds on the LLMail-Inject challenge, organized by Microsoft, ISTA and ETH Zurich: its
[code](https://github.com/microsoft/llmail-inject-challenge),
[dataset](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) and
[paper](https://arxiv.org/abs/2506.09956). The `challenge/` folder contains an unmodified subset of the original
code from commit `3da2399`, © Microsoft Corporation, under the MIT license ([`challenge/LICENSE`](challenge/LICENSE)).
This project is not affiliated with or endorsed by Microsoft.

```
@article{abdelnabi2025llmail,
  title     = {LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge},
  author    = {Sahar Abdelnabi and Aideen Fay and Ahmed Salem and Egor Zverev and Chi-Huang Liu and Chun-Chih Kuo and Jannis Weigend and Danyael Manlangit and Alex Apostolov and Haris Umair and João Donato and Masayuki Kawakita and Athar Mahboob and Tran Huu Bach and Tsun-Han Chiang and Myeongjin Cho and Hajin Choi and Byeonghyeon Kim and Hyeonjin Lee and Benjamin Pannell and Conor Mac Amhlaoibh and Mark Russinovich and Andrew Paverd and Giovanni Cherubin},
  journal   = {arXiv preprint arXiv:2506.09956},
  year      = {2025}
}
```

## License

The original challenge code in `challenge/` is MIT-licensed (© Microsoft Corporation). A license for the rest of
this repository has not been chosen yet.
