"""Serve the original challenge model, Phi-3-medium-128k-instruct (bf16), as an OpenAI-compatible endpoint on Modal.

Every setting below closes a way the replay could silently differ from computing each prompt on its own:
  - weights pinned to the Hugging Face revision that was live during the challenge (later commits only touch the
    README / a data card)
  - vLLM 0.8.5 + transformers 4.51.3: current transformers (5.x) can't load Phi-3's longrope config at all
  - one sequence at a time (--max-num-seqs 1): vLLM picks Phi-3's short/long RoPE scaling per *batch* (any token
    past position 4096 switches every token in the batch), so batching would let one request's length change
    another's numerics. 111 of the 480 Stage-1 prompts exceed 4096 tokens.
  - V0 engine with chunked prefill and prefix caching off: chunked prefill would rotate a long prompt's first chunk
    with short scaling and the rest with long; prefix caching would reuse short-scaled KV for the shared system
    prompt inside long prompts. (The V1 engine can't turn chunked prefill off.)
  - 32k context: the longest Stage-1 prompt is 23.6k Phi-3 tokens, plus up to 500 generated
  - bf16, no quantization; sampling comes from each request (replay.py sends top_p 0.92, temperature 1.0, seed)

Auth: the Modal URL is public, so vLLM checks a bearer key read from its VLLM_API_KEY environment variable, which the
Modal secret `phi3-vllm` provides. Don't pass it as `--api-key`: vLLM prints its CLI args, key included, in the startup
log. vLLM only checks the key on /v1/* routes (/health, /tokenize, /invocations etc. are open), so keep the URL to
yourself and stop the app when done. max_containers=1 so Modal never bills a second GPU.

Usage (see README.md in this folder):
    modal deploy modal_vllm.py        # prints the URL; the first start downloads ~28 GB into a Volume
    ... python replay.py ... --base-url https://<url>/v1 --api-key-env VLLM_API_KEY ...
    modal app stop phi3-medium-128k   # stop billing when done
"""
import subprocess

import modal

MODEL = "microsoft/Phi-3-medium-128k-instruct"
REVISION = "fa7d2aa4f5ea69b2e36b20d050cdae79c9bfbb3f"      # 2024-08-20, live through the challenge
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.8.5", "transformers==4.51.3")
    .env({"VLLM_USE_V1": "0"})                               # V0: chunked prefill can be disabled
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("phi3-medium-128k")


@app.server(
    image=image,
    gpu="L40S",                                  # 48 GB: 28 GB bf16 weights + KV for one 32k sequence
    secrets=[modal.Secret.from_name("phi3-vllm", required_keys=["VLLM_API_KEY"])],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    port=8000,
    startup_timeout=30 * MINUTES,                # first boot downloads the weights
    scaledown_window=10 * MINUTES,               # idle GPU shuts down 10 min after the last request
    max_containers=1,
    target_concurrency=4,
    unauthenticated=True,                        # auth is vLLM's bearer key (VLLM_API_KEY from the secret)
)
class Server:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm", "serve", MODEL,
            "--revision", REVISION,
            "--served-model-name", MODEL,
            "--host", "0.0.0.0", "--port", "8000",       # key comes from env VLLM_API_KEY (secret)
            "--dtype", "bfloat16",
            "--max-model-len", "32768",
            "--max-num-seqs", "1",
            "--no-enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--seed", "100",
        ]
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
