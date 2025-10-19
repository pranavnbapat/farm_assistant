# app/clients/hf_local_client.py

import os
import threading
import time

from typing import Dict, Any, Generator, Optional, AsyncGenerator

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer

# ---- Lazy singleton load (so the process starts quickly) --------------------
_MODEL = None
_TOKENIZER = None
_GENCFG = None

def _model_device():
    """Return the torch.device where the model's parameters live."""
    assert _MODEL is not None, "Model not initialised. Call _ensure_model() first."
    # some static analysers don’t see that _MODEL is a torch.nn.Module
    return next(_MODEL.parameters()).device  # type: ignore[union-attr]

def _ensure_model():
    """Load the chat model once. device_map='auto' will put it on GPU if present, otherwise CPU."""
    global _MODEL, _TOKENIZER, _GENCFG
    if _MODEL is not None:
        return

    # Allow overriding via env; default to DeepSeek chat model
    model_name = os.getenv("HF_CHAT_MODEL", "deepseek-ai/deepseek-llm-7b-chat")

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)

    # Decide where to load and offload weights
    dtype = torch.float32  # safest for CPU; or torch.bfloat16 on recent Xeons
    offload_dir = os.path.join("/tmp", "deepseek_offload")
    os.makedirs(offload_dir, exist_ok=True)

    # bfloat16 is fine on recent GPUs/CPUs; fall back to float16/32 if needed.
    # dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # or "cpu" if you prefer full CPU load
        offload_folder=offload_dir,  # <— add this line
        torch_dtype=dtype,  # torch_dtype is deprecated
        low_cpu_mem_usage=True,
    )
    _GENCFG = GenerationConfig.from_pretrained(model_name)
    # Ensure pad token exists (some chat models omit it)
    _GENCFG.pad_token_id = _GENCFG.eos_token_id

    assert _MODEL is not None and _TOKENIZER is not None and _GENCFG is not None

def _build_inputs_from_prompt(prompt: str) -> torch.Tensor:
    """
    Your build_prompt() returns a single string (system+rules+sources+question).
    If the tokenizer supports a chat template, we wrap it as a single user message.
    Otherwise we just encode the raw string.
    """
    # Always make sure model + tokenizer exist
    _ensure_model()

    global _TOKENIZER
    if _TOKENIZER is None:
        raise RuntimeError("Tokenizer failed to initialise")

    # If this tokenizer supports chat template (DeepSeek does), use that.
    if hasattr(_TOKENIZER, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return _TOKENIZER.apply_chat_template(  # type: ignore
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

    # Fallback: use the encode interface (works for every tokenizer)
    input_ids = torch.tensor(
        [_TOKENIZER.encode(prompt, add_special_tokens=True)]  # type: ignore
    )
    return input_ids


# ---- Public: drop-in API ----------------------------------------------------

def build_gen_payload(prompt: str, temperature: float, num_predict: int, *,
                      num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """
    Provided so your router can keep calling a similarly named function if needed.
    (Parity with ollama_client.build_gen_payload)
    """
    return {
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }

async def generate_once(prompt: str, temperature: float, num_predict: int,
                        model: str | None = None, num_ctx: int | None = None) -> str:
    """
    Non-streaming one-shot generation. Signature matches ollama_client.generate_once.
    """
    _ensure_model()
    assert _MODEL is not None and _TOKENIZER is not None

    inputs = _build_inputs_from_prompt(prompt)
    if num_ctx and inputs.shape[1] > num_ctx:
        inputs = inputs[:, -num_ctx:]

    inputs = inputs.to(_model_device())

    with torch.no_grad():
        out = _MODEL.generate(  # type: ignore[union-attr]
            inputs,
            max_new_tokens=int(num_predict),
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
            repetition_penalty=1.1,
        )

    assert _TOKENIZER is not None, "Tokenizer not initialised"  # helps IDE
    text = _TOKENIZER.decode(  # type: ignore[union-attr]
        out[0][inputs.shape[1]:],
        skip_special_tokens=True
    )
    return text.strip()

async def stream_generate(
    prompt: str,
    temperature: float,
    num_predict: int,
    context: list[int] | None = None,   # kept for API parity; not used
    model: str | None = None,           # kept for API parity; not used
    num_ctx: int | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming generation using TextIteratorStreamer.
    Parameters kept identical to Ollama for drop-in compatibility.
    Only prompt/temperature/num_predict/num_ctx are actually used.
    """
    _ensure_model()

    # ---- explicit guards so IDE knows they're real ----
    global _TOKENIZER, _MODEL
    assert _TOKENIZER is not None, "Tokenizer not initialised"
    assert _MODEL is not None, "Model not initialised"

    t0 = time.perf_counter()

    inputs = _build_inputs_from_prompt(prompt)
    if num_ctx and inputs.shape[1] > num_ctx:
        inputs = inputs[:, -num_ctx:]
    inputs = inputs.to(_model_device())

    # type: ignore silences the “could be None” hint for PyCharm
    streamer = TextIteratorStreamer(_TOKENIZER, skip_special_tokens=True, skip_prompt=True)  # type: ignore[arg-type]
    gen_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=int(num_predict),
        do_sample=True,
        temperature=float(temperature),
        top_p=0.9,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    # Start generation in a background thread
    th = threading.Thread(target=_MODEL.generate, kwargs=gen_kwargs)  # type: ignore[union-attr]
    th.start()

    token_count = 0
    for piece in streamer:
        token_count += 1
        yield {"response": piece}

    th.join()
    dt = max(1e-6, time.perf_counter() - t0)
    yield {
        "done": True,
        "done_reason": "stop",
        "eval_count": token_count,
        "eval_duration": dt,
        "tokens_per_sec": token_count / dt,
    }
