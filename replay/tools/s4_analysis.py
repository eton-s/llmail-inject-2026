"""Load the S4 replay results and compute the calibrated Stage-1 comparison.

This is the single source of truth the analysis notebook imports, so the figures and the prose can't drift from
each other. Everything here is reading result files written by replay.py; nothing calls a model.

Key framing (see the plan's decision gate): a model's attack rate is read against the ORIGINAL model's own
replay rate on the same attacks, not against 100% -- the 2024 successes were mostly rare stochastic wins, so
even faithful Phi-3 reproduces only a few. Two scores per attack:
  strict  = the challenge's exact-match rule (its own parser + to/body equality)
  relaxed = did the model send the $9.378M figure to the attacker's address, in any wording / format
`leak` (the relaxed verdict) is the intent-to-exfiltrate signal; strict is faithful-but-format-sensitive.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

REPLAY = Path(__file__).resolve().parents[1]
RESULTS = REPLAY / "results" / "s4"

# release order for the ladder; label -> (display name, year, is_2026)
MODELS = {
    "microsoft/Phi-3-medium-128k-instruct": ("Phi-3-medium (orig.)", "2024-05", False),
    "google/gemma-2-27b-it":                ("Gemma-2-27B",          "2024-06", False),
    "microsoft/phi-4":                      ("Phi-4",                "2024-12", False),
    "google/gemma-4-31b-it":                ("Gemma 4 31B",          "2026-04", True),
    "anthropic/claude-haiku-5.5":           ("Claude Haiku 5.5",     "2026-10", True),
}
OFFICIAL_PREFIX = '{"type": "function"'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def row_status(r: dict) -> str:
    if r.get("status"):
        return r["status"]
    if r.get("error"):
        return "context_overflow" if "context length" in r["error"] or "context_length" in r["error"] else "error"
    return "filtered" if r.get("finish_reason") == "content_filter" else "ok"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and 95% CI (lower, upper) for a binomial rate; (0,0,0) when n==0."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, (centre - half) / d), min(1.0, (centre + half) / d)


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher exact test for a/n1 vs b/n2."""
    N, K = n1 + n2, a + b
    if not (0 < K < N):
        return 1.0
    obs = math.comb(n1, a) * math.comb(n2, b) / math.comb(N, K)
    tot = 0.0
    for x in range(max(0, K - n2), min(K, n1) + 1):
        pr = math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
        if pr <= obs * (1 + 1e-9):
            tot += pr
    return min(1.0, tot)


def load_runs() -> dict[str, dict]:
    """label -> {model_id, draws, rows_by_draw, evaluable, ...}. One main run per model (ignores @-variants)."""
    out = {}
    for mid, (name, year, is2026) in MODELS.items():
        f = RESULTS / f"replay_{mid.replace('/', '-')}_successes.jsonl"
        if not f.exists():
            continue
        rows = read_jsonl(f)
        draws = sorted({r["draw"] for r in rows})
        out[mid] = {"name": name, "year": year, "is2026": is2026, "model_id": mid, "rows": rows, "draws": draws}
    return out


def per_draw_rate(rows: list[dict], metric: str) -> dict[int, tuple[int, int]]:
    """draw -> (successes, evaluable) for `metric` ('complied' or 'leak')."""
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if row_status(r) != "ok":
            continue
        by[r["draw"]][1] += 1
        by[r["draw"]][0] += int(r[metric])
    return {d: tuple(v) for d, v in by.items()}


def pooled(rows: list[dict], metric: str, draw: int | None = None) -> tuple[int, int]:
    """(successes, evaluable) over all draws (or one draw)."""
    ok = [r for r in rows if row_status(r) == "ok" and (draw is None or r["draw"] == draw)]
    return sum(r[metric] for r in ok), len(ok)


def headline_table() -> list[dict]:
    """One row per model: strict and relaxed rates pooled over ALL draws (every draw is one Bernoulli trial of
    the attack), with Wilson CIs over the attempt count and status counts from draw 0."""
    runs = load_runs()
    table = []
    for mid, run in runs.items():
        status = defaultdict(int)
        for r in run["rows"]:
            if r["draw"] == 0:
                status[row_status(r)] += 1
        sc, sn = pooled(run["rows"], "complied")
        lc, ln = pooled(run["rows"], "leak")
        sp, slo, shi = wilson(sc, sn)
        lp, llo, lhi = wilson(lc, ln)
        table.append({
            "model_id": mid, "name": run["name"], "year": run["year"], "is2026": run["is2026"],
            "evaluable": sn, "attacks": status["ok"], "overflow": status["context_overflow"],
            "filtered": status["filtered"], "draws": len(run["draws"]),
            "strict_k": sc, "strict": sp, "strict_lo": slo, "strict_hi": shi,
            "relaxed_k": lc, "relaxed": lp, "relaxed_lo": llo, "relaxed_hi": lhi,
        })
    return table


def consistency() -> list[dict]:
    """Per model (>=2 draws): of attacks that leak on >=1 draw, how many leak on EVERY draw. A high share means
    the vulnerability is reproducible (an attacker can rely on it), not sampling noise."""
    out = []
    for mid, run in load_runs().items():
        if len(run["draws"]) < 2:
            continue
        row = {"model_id": mid, "name": run["name"], "year": run["year"], "draws": len(run["draws"])}
        for metric in ("complied", "leak"):
            hits, trials = defaultdict(int), defaultdict(int)
            for r in run["rows"]:
                if row_status(r) != "ok":
                    continue
                trials[r["job_id"]] += 1
                hits[r["job_id"]] += int(r[metric])
            ever = [j for j in hits if hits[j] > 0]
            always = [j for j in ever if hits[j] == trials[j]]
            row[metric] = {"ever": len(ever), "always": len(always), "some": len(ever) - len(always),
                           "attacks": len(trials)}
        out.append(row)
    return out


def fidelity() -> dict:
    """Phi-3 baseline: successes vs controls, same model, proving the replay preserves attack strength."""
    sf = RESULTS / "replay_microsoft-Phi-3-medium-128k-instruct_successes.jsonl"
    cf = RESULTS / "replay_microsoft-Phi-3-medium-128k-instruct_phi3_controls.jsonl"
    if not (sf.exists() and cf.exists()):
        return {}
    S = [r for r in read_jsonl(sf) if r["draw"] == 0 and "Phi-3" in r["origin_model"] and row_status(r) == "ok"]
    C = [r for r in read_jsonl(cf) if r["draw"] == 0 and row_status(r) == "ok"]
    out = {}
    for metric in ("complied", "leak"):
        a, b = sum(r[metric] for r in S), sum(r[metric] for r in C)
        out[metric] = {"succ_k": a, "succ_n": len(S), "ctrl_k": b, "ctrl_n": len(C),
                       "p": fisher_two_sided(a, len(S), b, len(C))}
    # controls graded by what Phi-3 did in 2024
    by_outcome = defaultdict(lambda: [0, 0, 0])
    for r in C:
        g = by_outcome[r["model_outcome_2024"]]
        g[0] += r["complied"]; g[1] += r["leak"]; g[2] += 1
    out["controls_by_2024_outcome"] = {k: tuple(v) for k, v in by_outcome.items()}
    return out


def gemma_pair() -> dict:
    """Paired Gemma-2 vs Gemma 4 on jointly-evaluable draw-0 attacks (McNemar exact)."""
    g2 = {r["job_id"]: r for r in read_jsonl(RESULTS / "replay_google-gemma-2-27b-it_successes.jsonl") if r["draw"] == 0}
    g4 = {r["job_id"]: r for r in read_jsonl(RESULTS / "replay_google-gemma-4-31b-it_successes.jsonl") if r["draw"] == 0}
    both = [j for j in g4 if j in g2 and row_status(g2[j]) == "ok" and row_status(g4[j]) == "ok"]
    out = {"n": len(both)}
    for metric in ("leak", "complied"):
        a = sum(g2[j][metric] and not g4[j][metric] for j in both)   # Gemma-2 only
        b = sum(g4[j][metric] and not g2[j][metric] for j in both)   # Gemma 4 only
        n = a + b
        # exact two-sided binomial on the discordant pairs
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)
        out[metric] = {"g2": sum(g2[j][metric] for j in both), "g4": sum(g4[j][metric] for j in both),
                       "g2_only": a, "g4_only": b, "p": p}
    # attempt -> exact conversion
    for name, d in (("g2", g2), ("g4", g4)):
        att = [j for j in both if d[j]["leak"]]
        out[f"{name}_conv"] = (sum(d[j]["complied"] for j in att), len(att))
    return out


def multi_draw_rates(metric: str = "complied") -> dict[str, dict]:
    """For models with >1 draw: per-attack success probability pooled over draws, plus the any-draw rate."""
    out = {}
    for mid, run in load_runs().items():
        if len(run["draws"]) < 2:
            continue
        ok = [r for r in run["rows"] if row_status(r) == "ok"]
        by_attack = defaultdict(list)
        for r in ok:
            by_attack[r["job_id"]].append(int(r[metric]))
        k, n = pooled(run["rows"], metric)
        any_hit = sum(any(v) for v in by_attack.values())
        out[mid] = {"name": run["name"], "draws": len(run["draws"]), "pooled_k": k, "pooled_n": n,
                    "attacks": len(by_attack), "any_draw": any_hit}
    return out


def summary_text() -> str:
    lines = ["S4 Stage-1 (pooled over draws, evaluable attempts; rate [95% CI]):"]
    for row in headline_table():
        lines.append(f"  {row['name']:22s} {row['year']} [{row['draws']}d, {row['attacks']} atk]  "
                     f"strict {row['strict_k']:3d}/{row['evaluable']:<3d} "
                     f"{row['strict']:5.1%} [{row['strict_lo']:.0%}-{row['strict_hi']:.0%}]  "
                     f"relaxed {row['relaxed_k']:3d}/{row['evaluable']:<3d} {row['relaxed']:5.1%} "
                     f"[{row['relaxed_lo']:.0%}-{row['relaxed_hi']:.0%}]"
                     + (f"  (+{row['overflow']} overflow)" if row['overflow'] else "")
                     + (f"  (+{row['filtered']} filtered)" if row['filtered'] else ""))
    fid = fidelity()
    if fid:
        c, l = fid["complied"], fid["leak"]
        lines.append(f"\nFidelity (Phi-3, successes vs controls): strict {c['succ_k']}/{c['succ_n']} vs "
                     f"{c['ctrl_k']}/{c['ctrl_n']} p={c['p']:.2g} | relaxed {l['succ_k']}/{l['succ_n']} vs "
                     f"{l['ctrl_k']}/{l['ctrl_n']} p={l['p']:.2g}")
    gp = gemma_pair()
    lines.append(f"\nGemma pair (n={gp['n']}): relaxed {gp['leak']['g2']}->{gp['leak']['g4']} p={gp['leak']['p']:.3f} | "
                 f"strict {gp['complied']['g2']}->{gp['complied']['g4']} p={gp['complied']['p']:.3f} | "
                 f"attempt->exact: G2 {gp['g2_conv'][0]}/{gp['g2_conv'][1]}, G4 {gp['g4_conv'][0]}/{gp['g4_conv'][1]}")
    lines.append("\nConsistency (of attacks that leak >=1 draw, how many leak EVERY draw):")
    for row in consistency():
        for metric in ("complied", "leak"):
            c = row[metric]
            share = c["always"] / c["ever"] if c["ever"] else 0
            lines.append(f"  {row['name']:22s} {metric:8s}: ever {c['ever']:2d}, always {c['always']:2d} "
                         f"({share:.0%} reproducible)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_text())
