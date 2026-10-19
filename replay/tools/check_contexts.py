"""Independent byte-for-byte check of a contexts file against the challenge's own prompt-assembly code.

Extracts `LLMWithDefenses.prompt` and `Phi3LLM.call_model` from `challenge/agent/workloads/llm.py` (via `ast`, so
llm.py's torch/Azure imports never run), executes them unmodified with stub clients, captures the payload Phi-3
would have received for every record, and compares it with the record's `full_prompt`. The real `prompt_utils`
module supplies the Spotlight constants and `format_emails` (loaded under a stub package, so its relative import
of `models` resolves without running the agent package's __init__).

Usage:
    python tools/check_contexts.py [contexts/contexts_s4.jsonl]
"""
from __future__ import annotations

import ast
import importlib
import json
import logging
import sys
import types
from pathlib import Path

REPLAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY))
from build_contexts import DATASET, WORKLOADS, load_prompts  # noqa: E402
from replay import load_config  # noqa: E402

LLM_PY = WORKLOADS / "llm.py"
PHASE1_ALL = ["task_tracker", "prompt_shield", "llm_judge", "spotlight"]   # scenarios.LLMRouter, phase 1


def official_modules():
    """The real prompt_utils and models modules, loaded under a stub package so the agent package's __init__
    (torch/Azure) never runs; submodules still load from the real folder."""
    if "llmail_workloads" not in sys.modules:
        pkg = types.ModuleType("llmail_workloads")
        pkg.__path__ = [str(WORKLOADS)]
        sys.modules["llmail_workloads"] = pkg
    return (importlib.import_module("llmail_workloads.prompt_utils"),
            importlib.import_module("llmail_workloads.models"))


def official_functions():
    """The two llm.py methods that assemble Phi-3's prompt, bound to the real prompt_utils/models modules."""
    prompt_utils, models = official_modules()
    tree = ast.parse(LLM_PY.read_text(encoding="utf-8"))

    def method(cls: str, name: str) -> ast.FunctionDef:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == name:
                        return item
        raise LookupError(f"{cls}.{name} not found in {LLM_PY.name}")

    ns = {name: getattr(prompt_utils, name) for name in dir(prompt_utils) if name.startswith("SPOTLIGHT_")}
    ns.update(format_emails=prompt_utils.format_emails, parse_tool_calls=prompt_utils.parse_tool_calls,
              LLMResponse=models.LLMResponse, logging=logging, COMPETITON_PHASE="phase1")
    code = ast.Module(body=[method("LLMWithDefenses", "prompt"), method("Phi3LLM", "call_model")], type_ignores=[])
    exec(compile(code, str(LLM_PY), "exec"), ns)
    return ns["prompt"], ns["call_model"]


class CaptureClient:
    """Stands in for azure.ai.inference.ChatCompletionsClient: records the payload, returns an empty reply."""

    def complete(self, payload, seed=None):
        self.payload, self.seed = payload, seed
        message = types.SimpleNamespace(content="")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def first_diff(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPLAY / "contexts" / "contexts_s4.jsonl"
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    cfg = load_config()

    # the deployed strings, re-read from their source (not from the builder's output) + GeneralWorkload's renaming
    prompts = load_prompts(meta["prompt_source"], meta["scenario"], cfg, DATASET)
    send = cfg["llm_tools"][0]["function"]["name"]
    tool_prompt = prompts["tool"].replace(send, meta["tool_name"])
    meta_ok = meta["prompts"] == {**prompts, "tool": tool_prompt}

    prompt_fn, call_model = official_functions()
    client = CaptureClient()
    phi3 = types.SimpleNamespace(llm_system_prompt_tool=tool_prompt, max_new_tokens=cfg["max_new_tokens"],
                                 top_p=cfg["top_p"], open_source_client=client)
    phi3.call_model = types.MethodType(call_model, phi3)
    router = types.SimpleNamespace(system_prompt=prompts["system"], dry_run=False, llm_name="Phi3_api",
                                   llms={"Phi3_api": phi3},
                                   classifier=types.SimpleNamespace(run_detection_defense=lambda *a, **k: {}))

    n = same = 0
    bad = []
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        router.defenses = set(PHASE1_ALL) if rec["defense"] == "all" else {rec["defense"]}
        prompt_fn(router, prompts["query"], rec["context_emails"], rec["rank"])
        msgs = client.payload["messages"]
        official = msgs[0]["content"] if len(msgs) == 1 and msgs[0]["role"] == "user" else None
        n += 1
        if official == rec["full_prompt"]:
            same += 1
        else:
            bad.append((rec["job_id"], rec["level"], official, rec["full_prompt"]))

    print(f"{path.name}: {same}/{n} prompts byte-identical to the official llm.py assembly "
          f"(prompt source: {meta['prompt_source']}; meta strings match source: {meta_ok})")
    print(f"official Phi-3 payload settings: top_p={client.payload.get('top_p')} "
          f"max_tokens={client.payload.get('max_tokens')} seed={client.seed} temperature=<server default>")
    for job_id, level, official, ours in bad[:5]:
        if official is None:
            print(f"  DIFF {job_id} {level}: official payload was not a single user message")
            continue
        i = first_diff(official, ours)
        print(f"  DIFF {job_id} {level} at char {i}: official={official[i:i + 60]!r} ours={ours[i:i + 60]!r}")
    sys.exit(0 if same == n and meta_ok else 1)


if __name__ == "__main__":
    main()
