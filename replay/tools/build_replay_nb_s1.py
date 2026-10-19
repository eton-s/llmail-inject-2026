"""Build llmail-inject-2026/replay/replay_analysis.ipynb — summarizes the replay runs so far."""
import json, uuid, sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "replay_analysis.ipynb"

CELLS = [
    ("markdown", r'''# Replay results — attack transferability & defense decomposition

This notebook summarizes the `replay.py` runs. Each run takes the **scenario-1 attacks that fully succeeded against the original 2024 challenge stack** (Phi-3 / GPT-4o-mini + defenses, carried in `attack_subset_phase1.jsonl`) and replays them through a modern model, scoring whether the model emits the exact `send_email` tool call (**compliance**).

Two knobs per run:
- **model** — the assistant (Gemini, Phi-4, Gemma-2, ...).
- **defense** — `strict` = the challenge's anti-injection system prompt; `neutral` = that prompt ablated away (role + tool kept, hardening removed).

It auto-discovers every `replay/results/*.jsonl`, so it updates as you add runs.'''),

    ("markdown", r'''## Fidelity & deviations from the original challenge

This reuses the challenge's exact config, attacks, context assembly, input sanitization, tool-name randomization, and scoring — but it is **not** a reproduction of the original leaderboard. Key differences:

- **System prompt** — the `strict` prompt is the challenge's own `llm_system_prompt` (present in *every* original level); we did not add it. `neutral` ablates it (our analysis knob).
- **No external detectors** — the original ran each attack against a *level* = model **+ a defense** (Prompt Shield / TaskTracker / LLM-judge / Spotlight). We run the **model + base system prompt only** (`defense.undetected` is always True; we don't apply Spotlight's data-marking). So **`complied` measures model-only compliance, not the original model+defense "success."**
- **Different models / decoding** — Phi-4 / Gemini / Gemma vs the original Phi-3 / GPT-4o-mini; the Phi-3 Colab baseline is 4-bit + greedy (vs fp16 + `top_p=0.92` sampling). A bf16 vLLM run removes the quantization gap.
- **Tool delivery** — the Phi-3 json path matches the original Phi-3 handling; native-tool backends deliver the emails inline rather than as the original GPT `read_emails` tool-result.
- **Subset** — only attacks that *fully succeeded* against the 2024 stack (biased to 2024's winners); scenario 1 so far.

**Implication:** every model here is measured the same way, so **model-vs-model is fair** ("did the model get more robust?"). Scope claims to *model-only compliance under the challenge's system prompt* — not a reproduction of the challenge's success rates.'''),

    ("code", r'''import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

# palette (same tokens as analysis.ipynb Section 7)
BLUE, ORANGE, GRAY, LIGHT_GRAY = "#2a78d6", "#eb6834", "#c3c2b7", "#e1e0d9"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 110,
    "font.size": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 11,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": LIGHT_GRAY, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
    "xtick.color": GRAY, "ytick.color": GRAY, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2})

HERE = Path.cwd()
RESULTS = next((p for p in [HERE / "results", HERE / "replay" / "results"] if p.exists()), HERE / "results")
SUBSET = next((p for p in [Path("../../notebooks/attack_subsets/attack_subset_phase1.jsonl"),
                           Path(r"C:\Users\eton1\llmail-inject\notebooks\attack_subsets\attack_subset_phase1.jsonl")]
               if p.exists()), None)
print("results dir:", RESULTS.resolve())
print("subset     :", SUBSET.resolve() if SUBSET else "NOT FOUND")'''),

    ("code", r'''# --- load every result file into one long DataFrame (one row per attack per run) ---
FLAGS = ["email.retrieved", "defense.undetected", "exfil.sent", "exfil.destination", "exfil.content"]

def load_results(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*.jsonl")):
        system = "neutral" if "_neutral" in f.name else "strict"      # defense mode lives in the filename
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            rows.append({"file": f.name, "model": r.get("model"), "defense": system,
                         "job_id": r.get("job_id"), "complied": bool(r.get("complied")),
                         "error": bool(r.get("error")), "tool_called": bool(r.get("tool_calls"))})
    return pd.DataFrame(rows)

df = load_results(RESULTS)
# join the attack text (for technique tagging) from the subset
sub = {json.loads(l)["job_id"]: json.loads(l) for l in open(SUBSET, encoding="utf-8")} if SUBSET else {}
df["body"] = df["job_id"].map(lambda j: sub.get(j, {}).get("body", ""))
df["subject"] = df["job_id"].map(lambda j: sub.get(j, {}).get("subject", ""))
print(f"{len(df)} attack-runs across {df['file'].nunique()} files, "
      f"{df['model'].nunique()} models, defenses={sorted(df['defense'].unique())}")
df.head(3)'''),

    ("code", r'''# --- technique taxonomy (same detectors as analysis.ipynb Section 7) ---
TECH_PATTERNS = {
    "chat_template":      r"<\|?(?:im_start|im_end|system|user|assistant|end|endoftext)\|?>|\[/?INST\]",
    "fake_delimiter":     r"(?i)end[ _]?(?:of[ _]?)?(?:session|email|context|summary|conversation)|\[END|</?(?:email|context|summary|system|prompt|llm-note)>|NEW SESSION|ADMIN[_ ]?TERMINAL|######|======|-----",
    "ignore_previous":    r"(?i)ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|above|prior|preceding|earlier)|forget\s+(?:all|everything|previous)|disregard\s+(?:all|the|any|previous)",
    "json_api_format":    r'(?i)"\s*to\s*"\s*:|send_email|api_call|"\s*body\s*"\s*:|issue\s+(?:an?\s+)?api',
    "urgency":            r"(?i)\b(?:urgent|immediately|asap|critical|right away|as soon as possible|time[- ]sensitive)\b",
    "authority_persona":  r"(?i)\b(?:sudo|administrator|system message|you are (?:an?|now)|on behalf|supervisor|CEO|CFO|official instruction)\b",
}
TECH_LABELS = {"chat_template": "Chat-template tokens", "fake_delimiter": "Fake delimiters",
    "ignore_previous": "'Ignore previous'", "json_api_format": "JSON / tool scaffolding",
    "urgency": "Urgency", "authority_persona": "Authority / persona", "non_english_script": "Non-English script",
    "homoglyph_accent": "Homoglyphs", "word_gluing": "Word-gluing", "char_spacing": "Letter-spacing",
    "repetition": "Target repetition"}
TECHNIQUES = list(TECH_PATTERNS) + ["non_english_script", "homoglyph_accent", "word_gluing", "char_spacing", "repetition"]

def tag(text: pd.Series) -> pd.DataFrame:
    t = text.fillna(""); L = t.str.len().clip(lower=1); d = pd.DataFrame(index=t.index)
    for name, pat in TECH_PATTERNS.items():
        d[name] = t.str.contains(pat, regex=True, na=False)
    nonascii = t.str.count(r"[^\x00-\x7f]"); cjk = t.str.count(r"[\u3000-\u9fff\uac00-\ud7ff]")
    d["non_english_script"] = (cjk / L) > 0.05
    d["homoglyph_accent"] = ((nonascii - cjk) / L > 0.02) & ~d["non_english_script"]
    d["word_gluing"] = t.str.contains(r"[A-Za-z]{40,}", regex=True, na=False)
    d["char_spacing"] = t.str.count(r" \w ") >= 15
    d["repetition"] = t.str.count(r"(?i)contact@contact\.com|confirmation|\$NUM") >= 5
    return d[TECHNIQUES]

df = pd.concat([df, tag(df["body"])], axis=1)
print("tagged techniques:", ", ".join(TECHNIQUES))'''),

    ("markdown", r'''## 1. Compliance by model and defense

**Compliance rate** = share of attacks where the model emitted the exact `send_email` call. The denominator excludes API errors (503/429/etc.). These are 50-attack pilots unless a run used the full subset — small N, so read them as directional.'''),

    ("code", r'''g = df.groupby(["model", "defense"])
summary = pd.DataFrame({"attacks": g.size(), "errors": g["error"].sum(), "complied": g["complied"].sum()})
summary["valid"] = summary["attacks"] - summary["errors"]
summary["compliance rate"] = summary["complied"] / summary["valid"].clip(lower=1)
summary = summary[["attacks", "errors", "valid", "complied", "compliance rate"]].reset_index().sort_values(["model", "defense"])
display(summary.style.format({"compliance rate": "{:.1%}"})
        .background_gradient(cmap="Blues", subset=["compliance rate"]).hide(axis="index"))'''),

    ("code", r'''# grouped bars: compliance rate per model, strict vs neutral
piv = summary.pivot_table(index="model", columns="defense", values="compliance rate")
for col in ["strict", "neutral"]:
    if col not in piv:
        piv[col] = np.nan
piv = piv[["strict", "neutral"]].sort_index()
y = np.arange(len(piv)); h = 0.38
fig, ax = plt.subplots(figsize=(9, 0.8 + 0.7 * len(piv)))
ax.barh(y + h/2, piv["strict"].fillna(0), height=h, color=LIGHT_GRAY, label="strict (defense prompt on)")
ax.barh(y - h/2, piv["neutral"].fillna(0), height=h, color=BLUE, label="neutral (defense ablated)")
for yi, (s, n) in enumerate(zip(piv["strict"], piv["neutral"])):
    if not np.isnan(s): ax.text(s + 0.004, yi + h/2, f"{s:.0%}", va="center", fontsize=8.5, color=INK_2)
    if not np.isnan(n): ax.text(n + 0.004, yi - h/2, f"{n:.0%}", va="center", fontsize=8.5, color=INK_2)
ax.set_yticks(y, piv.index); ax.set_xlim(0, max(0.05, np.nanmax(piv.values) * 1.25))
ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
ax.set_title("Compliance on 2024's winning attacks: every model resists with the\ndefense prompt on; only the ablation lets any through")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout(); plt.show()'''),

    ("markdown", r'''## 2. What transfers: techniques among attacks that still complied

Among the attacks that got the model to comply (mostly the `neutral` runs), which techniques are over-represented versus the full attack set? This hints at *which kinds of attack survive* a modern model intrinsically. **Caveat: only a handful of attacks comply so far (N is tiny) — read this as suggestive, and confirm on the full subset before drawing conclusions.**'''),

    ("code", r'''complied = df[df["complied"]]
if len(complied):
    base = df.drop_duplicates("job_id")[TECHNIQUES].mean()      # prevalence across the attack set
    among = complied[TECHNIQUES].mean()                          # prevalence among compliant attacks
    comp = (pd.DataFrame({"among compliant": among, "in attack set": base})
            .rename(index=TECH_LABELS).sort_values("among compliant", ascending=False))
    display(comp.style.format("{:.0%}").background_gradient(cmap="Blues", subset=["among compliant"]))

    top = comp.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(np.arange(len(top)) + 0.2, top["among compliant"], height=0.38, color=BLUE, label="among compliant attacks")
    ax.barh(np.arange(len(top)) - 0.2, top["in attack set"], height=0.38, color=LIGHT_GRAY, label="in the attack set")
    ax.set_yticks(np.arange(len(top)), top.index); ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
    ax.set_title(f"Techniques in the {len(complied)} attacks that still complied")
    ax.legend(loc="lower right", frameon=False); plt.tight_layout(); plt.show()
else:
    print("No complied attacks yet — run a --system neutral pilot to populate this.")'''),

    ("code", r'''# the actual attacks that got through (text + which techniques), for inspection
if len(complied):
    for _, r in complied.drop_duplicates("job_id").iterrows():
        techs = [TECH_LABELS[t] for t in TECHNIQUES if r[t]]
        body = " ".join(str(r["body"]).split())
        print(f"[{r['model']} / {r['defense']}] tags: {', '.join(techs) or 'none'}")
        print(f"   {body[:200]}\n")'''),

    ("markdown", r'''## 3. Findings so far

- **The system prompt is the dominant scenario-1 defense.** With it on (`strict`), every model tested — Phi-4, Gemini 3.8, and even 2024-era Gemma-2-27B — sits at ~0% compliance. Model generation barely matters here.
- **Attacks mostly don't transfer even with the defense off.** Ablated (`neutral`), Phi-4 still blocked ~88% of attacks that fully beat the 2024 stack.
- **The attacks that do get through read as plausible task requests** ("Dear assistant, please send a confirmation…") and skew toward **authority/persona** framing (over-represented vs the attack set); the pure "ignore previous" / urgency / JSON-scaffolding tricks don't appear among them. With only ~6 compliant attacks this is suggestive, not conclusive.
- **Caveats:** 50-attack pilots (very noisy); Gemini's run lost 13 calls to 503s (pre-retry); compliance is measured model-alone (no external detector); each backend uses its model's natural tool interface (native tool-calling for Phi-4/Gemini, JSON-prompt for open models).

**Next:** ablation across the ladder (`neutral` runs for Gemma-2, Phi-4-mini, Gemini) to look for a real model/scale gradient; re-run Gemini (503-retry is in now); scale the key configs to the full ~1,530-attack subset; optionally add scenario 2.'''),
]

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in CELLS:
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    nb["cells"].append(cell)
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False); f.write("\n")
print(f"wrote {len(nb['cells'])} cells -> {NB_PATH}")
