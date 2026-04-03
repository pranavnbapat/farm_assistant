# Capacity Planning Notes

This document captures the current token-budget and concurrency picture for the Farm Assistant stack based on:

- the FastAPI app configuration in this repository
- the current `.env` values used by the app
- the vLLM supervisor configuration provided for the model server

It is intended as an operational planning note, not a theoretical model card.

## Inputs Used

### FastAPI App Settings

From [`.env`](/home/pranav/PyCharm/EU-FarmBook/farm_assistant/.env):

- `MAX_ACTIVE_GENERATIONS=3`
- `NUM_CTX=16384`
- `MAX_OUTPUT_TOKENS=1024`
- `MAX_INPUT_TOKENS=12000`
- `MAX_USER_INPUT_TOKENS=2000`
- `MAX_CONTEXT_CHARS=24000`
- `TOP_K=5`

From code:

- generation concurrency gate is created in [app/main.py](/home/pranav/PyCharm/EU-FarmBook/farm_assistant/app/main.py)
- input/output token budgeting is enforced in [app/routers/ask.py](/home/pranav/PyCharm/EU-FarmBook/farm_assistant/app/routers/ask.py)

### vLLM Server Settings

From the provided `supervisord.conf`:

- model: `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ`
- GPU: `A40 48 GB`
- `--gpu-memory-utilization 0.80`
- `--max-model-len 65536`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 65536`
- `CUDA_VISIBLE_DEVICES=0`

## What The App Actually Allows Per Chat Request

Although the model server is configured with `--max-model-len 65536`, the FastAPI app does not use that full window.

The app computes the prompt-side cap as:

```text
prompt_cap = min(MAX_INPUT_TOKENS, NUM_CTX - max_output_tokens - 256)
```

With the current values:

```text
NUM_CTX = 16384
MAX_OUTPUT_TOKENS = 1024

NUM_CTX - MAX_OUTPUT_TOKENS - 256
= 16384 - 1024 - 256
= 15104

prompt_cap = min(12000, 15104) = 12000
```

So the current effective per-request chat budget is:

- prompt/input side: up to about `12000` tokens
- output side: up to `1024` tokens
- total effective request budget: about `13024` tokens

This means the Farm Assistant app is currently operating as a roughly `13k-token` application, not a `32k` or `64k` application.

## Practical Input Size Estimate

For normal retrieval-backed chat turns, the input budget is typically made of:

- system/prompt instructions
- user message
- conversation history
- retrieved source context
- profile/facts context

### Typical Turn

Reasonable operational estimate for a normal turn:

- prompt instructions: `500-1500`
- user question: `10-200`
- history: `200-2000`
- profile/facts: `50-300`
- retrieved context: `4000-7000`

Typical total input:

- about `5000-8000` tokens

### Heavy Turn

If the chat history is longer and retrieval returns larger context:

- about `9000-12000` tokens

### Lightweight Conversational Turn

If the request routes to `conversation_only`, `history_only`, or `assistant_capabilities`:

- about `1000-3000` input tokens is a more realistic range

## What `MAX_ACTIVE_GENERATIONS=3` Actually Means

`MAX_ACTIVE_GENERATIONS=3` does not mean three users total.

It means:

- the FastAPI app will allow up to three concurrent generation requests into its own request pipeline
- a fourth request will wait in the app-level queue

That app-level limit is only one layer.

## The Real Hard Limit Right Now

Your current vLLM server is configured with:

```text
--max-num-seqs 1
```

That is the strongest concurrency constraint in the system.

It means:

- the text model server is currently configured to process only one active sequence at a time
- even if the FastAPI app allows three concurrent generation requests, the model server itself is effectively single-sequence

Operationally, this means:

- request 1 can generate
- requests 2 and 3 can queue upstream
- the vLLM server itself will not serve more than one active sequence concurrently on that instance

So with the current model-server configuration, your practical active generation concurrency is:

- `1` active generation on the text model server

not `3`

## GPU Memory Budget

You provided:

- GPU: `A40 48 GB`
- `--gpu-memory-utilization 0.80`

So the configured vLLM working budget is approximately:

```text
48 GB × 0.80 = 38.4 GB
```

That `38.4 GB` must cover:

- loaded model weights
- runtime buffers
- KV cache
- fragmentation and execution overhead

This is why context size and concurrency trade off directly against one another.

## What We Can Say Reliably Right Now

Based on the app and server settings you shared:

1. The app is currently budgeted for about `12k` prompt tokens and `1k` output tokens per chat request.
2. Typical retrieval-backed requests likely land around `5k-8k` input tokens.
3. Heavier requests can approach the `12k` prompt cap.
4. The app-level queue allows `3` concurrent generation requests to enter the app.
5. The model server is currently configured for only `1` active sequence with `--max-num-seqs 1`.

So the most accurate current statement is:

- the Farm Assistant UI can queue up to three concurrent generation requests
- but the underlying text model server is effectively single-generation right now

## Recommended Interpretation For Capacity

If you keep the current vLLM config:

- safe active generations: `1`
- additional simultaneous users: queued

If you want true parallel serving, `--max-num-seqs` is the first thing to revisit.

## Recommended Next Tuning Order

1. Keep the app-level request gate conservative until the model server is tuned.
2. Increase `--max-num-seqs` on vLLM before increasing `MAX_ACTIVE_GENERATIONS`.
3. Measure actual memory usage and throughput under load before changing `NUM_CTX`.
4. Only increase app-level concurrency after the model server demonstrates stable parallel execution.

## What Is Still Needed For A More Accurate Capacity Estimate

The current estimate is accurate enough for planning, but not enough for a precise safe-concurrency target above `1`.

To improve it, the most useful additional data would be:

1. vLLM startup log lines showing:
   - model weight memory
   - KV cache allocation
   - reported max concurrency / cache blocks

2. `nvidia-smi` while the model is loaded and while one generation is active:
   - memory used
   - GPU utilization

3. a short load test with:
   - one user
   - two concurrent users
   - three concurrent users
   and observed latency / failures / queueing

## Immediate Operational Conclusion

With the current settings, the honest answer is:

- the app behaves like a `~13k token` chat application
- the model server is currently configured to serve one active generation at a time

So if your question is "how many users can we serve in parallel right now?", the most accurate answer from the provided configuration is:

- `1` active model generation
- more users can connect and queue, but not generate in parallel on that vLLM instance

## Recommended Settings For Your Current Server

Server inputs provided:

- GPU: `A40 48 GB`
- model: `Qwen3-30B-A3B-Instruct-2507-AWQ`
- current vLLM:
  - `--gpu-memory-utilization 0.80`
  - `--max-model-len 65536`
  - `--max-num-seqs 1`
  - `--max-num-batched-tokens 65536`

### Option A: Keep Current vLLM Concurrency

If you keep:

- `--max-num-seqs 1`

then the honest Farm Assistant app config is:

```env
MAX_ACTIVE_GENERATIONS=1
NUM_CTX=16384
MAX_OUTPUT_TOKENS=1024
MAX_INPUT_TOKENS=12288
MAX_USER_INPUT_TOKENS=2000
MAX_CONTEXT_CHARS=37056
TOP_K=5
MAX_TOKENS=1024
```

Reason:

- this removes misleading extra queue depth in the app layer
- one active generation on vLLM matches one active generation admitted by FastAPI

### Option B: Recommended Balanced 2-User Parallel Target

If your goal is real parallel serving for two active generations, this is the recommended first step.

Suggested vLLM changes:

```text
--gpu-memory-utilization 0.85
--max-model-len 13568
--max-num-seqs 2
--max-num-batched-tokens 27136
```

Suggested Farm Assistant env:

```env
MAX_ACTIVE_GENERATIONS=2
NUM_CTX=12288
MAX_OUTPUT_TOKENS=1024
MAX_INPUT_TOKENS=8960
MAX_USER_INPUT_TOKENS=2000
MAX_CONTEXT_CHARS=23744
TOP_K=5
MAX_TOKENS=1024
```

Why this is the recommended balance:

- keeps the app below the current `16k` working-context habit
- preserves strong retrieval quality
- enables true two-sequence parallelism instead of pure queueing
- avoids jumping immediately to a more aggressive 3-way parallel setup on a single A40

### Option C: More Aggressive 3-User Target

This is possible as a tuning candidate, not the default recommendation.

Suggested vLLM starting point:

```text
--gpu-memory-utilization 0.88
--max-model-len 9472
--max-num-seqs 3
--max-num-batched-tokens 28416
```

Suggested Farm Assistant env:

```env
MAX_ACTIVE_GENERATIONS=3
NUM_CTX=8192
MAX_OUTPUT_TOKENS=1024
MAX_INPUT_TOKENS=5632
MAX_USER_INPUT_TOKENS=1408
MAX_CONTEXT_CHARS=12800
TOP_K=4
MAX_TOKENS=1024
```

Tradeoff:

- better parallelism
- lower retrieval/history budget per request
- more likely to need prompt/context trimming
- should be load-tested before production use

## Planner Script

A calculator script is included here:

- [testing/calculate_capacity.py](/home/pranav/PyCharm/EU-FarmBook/farm_assistant/testing/calculate_capacity.py)

It takes:

- GPU size
- vLLM memory utilization
- `max-model-len`
- `max-num-seqs`
- `max-num-batched-tokens`
- target active users
- desired output-token cap
- desired user-input cap

and produces:

- recommended Farm Assistant env values
- suggested vLLM flag values
- warnings when the target concurrency exceeds the current vLLM sequence limit
- optional in-place update of a Farm Assistant `.env` file

### Example: Your Current Server

```bash
python3 testing/calculate_capacity.py \
  --gpu-vram-gb 48 \
  --gpu-memory-utilization 0.80 \
  --vllm-max-model-len 65536 \
  --vllm-max-num-seqs 1 \
  --vllm-max-num-batched-tokens 65536 \
  --target-active-users 1
```

### Example: Planning For 2 Active Users

```bash
python3 testing/calculate_capacity.py \
  --gpu-vram-gb 48 \
  --gpu-memory-utilization 0.85 \
  --vllm-max-model-len 16384 \
  --vllm-max-num-seqs 2 \
  --vllm-max-num-batched-tokens 32768 \
  --target-active-users 2 \
  --format env
```

### Example: Write Recommended Values Into `.env`

```bash
python3 testing/calculate_capacity.py \
  --gpu-vram-gb 48 \
  --gpu-memory-utilization 0.85 \
  --vllm-max-model-len 16384 \
  --vllm-max-num-seqs 2 \
  --vllm-max-num-batched-tokens 32768 \
  --target-active-users 2 \
  --write-env-file .env \
  --format env
```

## What I Still Need From You For Better Accuracy

The script and recommendations above are the best possible from configuration alone. To tighten them further, the most useful missing inputs are:

1. `nvidia-smi` output:
   - idle after model load
   - during one active generation

2. vLLM startup log lines that mention:
   - memory profiling
   - KV cache allocation
   - max concurrency estimate

3. one small load test:
   - 1 active generation
   - 2 active generations
   - 3 active generations
   with observed latency and failure/queue behavior
