# Phi-3-medium-128k baseline on Modal

Serves the original challenge model (bf16, weights pinned to the revision live during the challenge) as an
OpenAI-compatible endpoint, then replays the Stage-1 prompts through `replay.py`. `modal_vllm.py` explains every
fidelity setting; the short version: one request at a time, no chunked prefill, no prefix caching, so each prompt
is computed exactly as if it were alone (Phi-3's long-context RoPE scaling is chosen per batch in vLLM).

Run everything from `llmail-inject-2026/replay/` with the venv active (Git Bash).

## 1. One-time setup
1. Create a free Modal account (Starter plan) at modal.com.
2. `modal` is already in the venv (`pip install modal` otherwise). Log in, which opens the browser:
   ```
   modal setup
   ```
3. Create the endpoint key and store it as a Modal secret. The key stays in this shell; if you lose it, generate
   a new one and re-run the second command with `--force`:
   ```
   export VLLM_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
   modal secret create phi3-vllm VLLM_API_KEY="$VLLM_API_KEY"
   ```

## 2. Deploy
```
modal deploy serve_phi3/modal_vllm.py
```
This prints the server URL (e.g. `https://<workspace>--phi3-medium-128k-server.us-east.modal.direct`).
The first start builds the image and downloads ~28 GB of weights into the `huggingface-cache` Volume
(allow 15-30 min). Later starts take a few minutes. Watch progress in the Modal dashboard.

Smoke test. It may take a while on a cold start; `no upstreams available` means it is still loading (watch
`modal app logs phi3-medium-128k` for `Application startup complete`). With the key it should list the model;
without the key it must return 401:
```
curl -i -H "Authorization: Bearer $VLLM_API_KEY" https://<url>/v1/models
curl -i https://<url>/v1/models
```
Security notes: vLLM checks the key only on `/v1/*` routes (`/invocations`, `/tokenize` etc. are open), so keep the
URL to yourself and stop the app when finished. If the key ever appears in a log or chat, rotate it: generate a new
one, re-run `modal secret create ... --force`, and redeploy.

## 3. Run
Set the base URL once:
```
export PHI3_URL=https://<url>/v1
```
Positive control first (expect the tool to fire; this is Phi-3's ceiling under the original sampling):
```
python replay.py --scenario 4 --model openai --model-id microsoft/Phi-3-medium-128k-instruct --base-url $PHI3_URL --api-key-env VLLM_API_KEY --sanity --draws 10
```
Then the 212 successes and the 268 Phi-3 controls. Requests run one at a time by design (about 10 s each);
`--concurrency 2` keeps the GPU busy. Both commands resume if interrupted.
```
python replay.py --scenario 4 --model openai --model-id microsoft/Phi-3-medium-128k-instruct --base-url $PHI3_URL --api-key-env VLLM_API_KEY --concurrency 2
python replay.py --scenario 4 --select phi3_controls --model openai --model-id microsoft/Phi-3-medium-128k-instruct --base-url $PHI3_URL --api-key-env VLLM_API_KEY --concurrency 2
```

## 4. Stop billing
The GPU shuts down 10 min after the last request, but stop the app explicitly when done:
```
modal app stop phi3-medium-128k
modal app list
```

**Cost:** L40S runs at about $2/hr (third-party reports; check modal.com/pricing). The whole Stage-1 baseline,
including startup, is roughly 1.5-2 GPU-hours, which should fit within the free Starter credits.
