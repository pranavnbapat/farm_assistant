# Farm Assistant Arena Model Configuration Findings

Date: 2026-06-19

This document summarises the current Farm Assistant Arena model/configuration information for the first Arena experiment. It separates what should be shown in the admin dashboard from what should be retained for reports and reproducibility.

## Recommended Metadata Levels

| Metadata item | Dashboard | Report/export | Notes |
|---|---:|---:|---|
| Variant id / backend key | Required | Required | Needed to map Answer A/B/C to the actual configuration. |
| Visible or hidden benchmark | Required | Required | OpenAI and Claude are hidden benchmarks, not participant-facing answers. |
| Model identifier/version | Required | Required | Minimum reproducibility field. |
| Provider and hosting environment | Required | Required | Important because model behavior may depend on runtime and deployment. |
| Generation settings | Required | Required | Temperature, top_p, max output tokens, and provider-specific token parameter names affect answer style. |
| Retrieval endpoint and retrieval count | Required | Required | Arena evaluates full RAG configurations, not only base models. |
| Score thresholds/filtering | Required | Required | Affects whether answers are grounded or fallback/general. |
| Prompt/citation/grounding rules | Summary | Required | Dashboard can show prompt family/version; report should explain rules. |
| Runtime latency/errors/token usage | Required | Required | Already captured per answer where providers return it. |
| Context length / input caps | Useful | Required | Necessary for fair interpretation, but not always known exactly. |
| Exact checkpoint/revision | Optional if unknown | Required if available | Some hosted providers expose only the model alias. |
| Quantisation format | Optional if unknown | Required if available | Often unknown for managed APIs. |
| Language-handling behavior | Useful | Required | Especially important for multilingual Arena questions. |
| Planned improvements | No | Optional appendix | Not per-run metadata. |

## Current Arena Variants

| Arena role | Variant/backend | Participant-visible? | Main purpose |
|---|---|---:|---|
| Current chatbot / v1 | `mistral` / `euf_chatbot_tnods` | Yes | Existing production baseline. |
| New Farm Assistant / v2 | `qwen3` / `um_qwen3` | Yes | Current Farm Assistant configuration using Qwen3. |
| EU model candidate | `eurollm` / `eurollm` | Yes | EU-based model candidate using the Farm Assistant RAG stack. |
| Proprietary benchmark | `openai` / `openai_gpt` | No | Hidden internal benchmark, stored but not shown to participants. |
| Proprietary benchmark | `anthropic` / `anthropic_claude` | No | Hidden internal benchmark, stored but not shown to participants. |

## Model and Runtime Configuration

| Variant | Model family | Model identifier | Size / revision | Provider and hosting | Runtime notes | Missing or assumed |
|---|---|---|---|---|---|---|
| Mistral current chatbot | Mistral Small | `azure_ai/mistral-small-2503` | Believed to be Mistral Small 3.1 24B; checkpoint `mistral-small-2503` | Mistral hosted through Azure AI endpoint; Azure AI auth; three Azure AI endpoints with LiteLLM simple-shuffle routing and 3 retries | Ollama fallback is present but commented out/disabled | Parameter size not explicitly confirmed; exact Foundry context setting unavailable; quantisation assumed fp16 |
| Qwen3 | Qwen3 | `qwen3-30b-a3b-awq` | 30B hint; AWQ quantised | Farm Assistant vLLM/OpenAI-compatible deployment | Current Farm Assistant RAG path, SSE streaming | Exact checkpoint/revision not confirmed beyond configured model id |
| EuroLLM | EuroLLM | Served as `current`; root model observed as `utter-project/EuroLLM-9B-Instruct-2512` | 9B; max model length observed as 16384 | RunPod vLLM/OpenAI-compatible endpoint | Uses the same Farm Assistant RAG path as Qwen3 | `.env.eurollm` still contains placeholder model/URL values locally; final deployed values should be recorded |
| OpenAI hidden benchmark | GPT | `gpt-5.4-mini` configured | Mini variant | Farm Assistant instance calling OpenAI Chat Completions-compatible API | Hidden benchmark; stored under same run with `is_benchmark=True` | Exact provider revision/context length should be confirmed from OpenAI account/docs |
| Claude hidden benchmark | Claude Haiku | `claude-haiku-4-5` configured | Haiku variant | Farm Assistant instance using Anthropic Messages API | Hidden benchmark; stored under same run with `is_benchmark=True` | Exact provider revision/context length should be confirmed from Anthropic account/docs |

## Generation Settings

| Variant | Temperature | top_p | Max output tokens | Input/context caps | Provider-specific behavior |
|---|---:|---:|---:|---|---|
| Mistral current chatbot | `0.2` | Default / not set | Default / not set | Context length typically `32k`, not confirmed in Foundry | LiteLLM routes across three Azure AI endpoints. |
| Qwen3 | `0.4` | `0.9` | `4000` configured | `NUM_CTX=16384`, `MAX_INPUT_TOKENS=12000`, `MAX_USER_INPUT_TOKENS=2000` | OpenAI-compatible vLLM streaming with usage requested via `stream_options.include_usage`. |
| EuroLLM | `0.4` | `0.9` | `4000` configured | `NUM_CTX=16384`, `MAX_INPUT_TOKENS=12000`, `MAX_USER_INPUT_TOKENS=2000` | OpenAI-compatible vLLM streaming; public vLLM `/v1/models` reported `max_model_len=16384`. |
| OpenAI hidden benchmark | Config says `0.4`, but GPT-5-style code omits custom temperature/top_p | Not sent for GPT-5-style models | `4000` configured | `NUM_CTX=16384`, `MAX_INPUT_TOKENS=12000`, `MAX_USER_INPUT_TOKENS=2000` | Code uses `max_completion_tokens` for `gpt-5*` models and avoids custom temperature/top_p. |
| Claude hidden benchmark | `0.4` | Not set | `4000` configured through `ANTHROPIC_MAX_TOKENS` / request max tokens | `NUM_CTX=16384`, `MAX_INPUT_TOKENS=12000`, `MAX_USER_INPUT_TOKENS=2000` | Farm Assistant converts OpenAI-style system messages into Anthropic top-level `system` plus user/assistant messages. |

## Retrieval and RAG Configuration

| Variant | Retrieval endpoint | Requested/candidate results | Score threshold | Context selection | Reranking/post-processing |
|---|---|---:|---:|---|---|
| Mistral current chatbot | `neural_search_relevant_new` | 3 results requested | KO score `>= 0.4`; no chunk-level threshold | Only chunks from KOs passing the KO threshold; top 2 passing chunks included in RAG context | Sort `score_desc`; no explicit reranking; `post_process_text()` strips `[[ ]]` to `[ ]` and removes `<p>` / `</p>` tags |
| Qwen3 | `/llm_retrieve` | Candidate `k=10`; final sources capped by context builder and token budget | `RETRIEVAL_MIN_SCORE=1.0` | Builds source blocks from title/subtitle/description/project metadata and `llm_context` or ranked `ko_content_flat`; max context chars `24000`; per-parent cap `3500` | No explicit neural reranker in Farm Assistant; paragraph ranking uses lexical overlap/metadata boosts when flat content is available |
| EuroLLM | `/llm_retrieve` | Candidate `k=10` | `RETRIEVAL_MIN_SCORE=1.0` | Same Farm Assistant context builder as Qwen3 | Same as Qwen3 |
| OpenAI hidden benchmark | `/llm_retrieve` through Farm Assistant instance | Candidate `k=10` | `RETRIEVAL_MIN_SCORE=1.0` | Same Farm Assistant context builder as Qwen3 | Same as Qwen3 |
| Claude hidden benchmark | `/llm_retrieve` through Farm Assistant instance | Candidate `k=10` | `RETRIEVAL_MIN_SCORE=1.0` | Same Farm Assistant context builder as Qwen3 | Same as Qwen3 |

## KO Fields and Source Material

| Variant group | KO/source fields supplied to model |
|---|---|
| Mistral current chatbot | `title`, `_id`, `_score`, `ko_content_scored[].text`; citation URLs built as `https://www.eufarmbook.eu/en/contributions/{farmbook_id}` |
| Farm Assistant variants: Qwen3, EuroLLM, OpenAI, Claude | Source block can include title, subtitle, description, project display/name/acronym/type, license, keywords, topics, themes, languages, creators, date of completion, `llm_context`, `ko_content_flat`, source id/url/display url, and source score. |

## Citation, Grounding, and Prompt Rules

| Variant | Citation format | Grounding behavior | Fallback behavior |
|---|---|---|---|
| Mistral current chatbot | Markdown `[N](url)` links, where `N` is source position and URL points to EU-FarmBook contribution page | Prompt says to keep answer grounded in `CONTEXT INFORMATION` and use exact links from prompt | If no documents contain the answer, model is instructed to provide a short introduction to retrieved documents because the user may still be interested |
| Qwen3 | Inline `[1]`, `[2]`, etc., matching source block numbers | Prompt says context is primary grounding material and citations must only use numbers present in the source block | If no EU-FarmBook source is found, say so and give cautious general agricultural answer only for general agriculture questions; do not cite fallback answer |
| EuroLLM | Same as Qwen3 | Same Farm Assistant prompt/citation rules | Same as Qwen3 |
| OpenAI hidden benchmark | Same as Qwen3 when run as Farm Assistant hidden variant | Same Farm Assistant prompt/citation rules | Same as Qwen3 |
| Claude hidden benchmark | Same as Qwen3 when run as Farm Assistant hidden variant | Same Farm Assistant prompt/citation rules | Same as Qwen3 |

## Language Handling

| Variant | Language behavior |
|---|---|
| Mistral current chatbot | Current production has a 10-language cap: English, French, German, Italian, Spanish, Polish, Dutch, Romanian, Greek, Greek Latin. Uses MediaPipe language detection with confidence threshold `0.7`; non-English input translated to English internally with up to 5 retries; output translated back only for supported detected languages; unsupported or low-confidence language detection may trigger an English warning; messages under 3 words bypass detection and default to English. |
| Qwen3 | Farm Assistant prompt instructs the model to answer in the language of the user question. Retrieved sources may be in other languages and should be used/translated as needed. No fixed 10-language cap in this implementation. |
| EuroLLM | Same Farm Assistant language instruction as Qwen3. |
| OpenAI hidden benchmark | Same Farm Assistant language instruction as Qwen3 when run through Farm Assistant hidden variant. |
| Claude hidden benchmark | Same Farm Assistant language instruction as Qwen3 when run through Farm Assistant hidden variant. |

## Runtime Data Already Captured

| Captured item | Where it is stored |
|---|---|
| Question and run id | `ChatComparisonRun.question`, `ChatComparisonRun.run_uuid` |
| Participant user | `ChatComparisonRun.user` |
| Participant expertise/role | `ChatComparisonRun.question_metadata.participant_profile` |
| Sample question/language | `ChatComparisonRun.question_metadata.sample_question` |
| Answer text | `ChatComparisonAnswer.assistant_message` |
| Variant id/backend/display label | `ChatComparisonAnswer.variant_id_snapshot`, `backend_key_snapshot`, `display_label` |
| Visible vs hidden benchmark | `ChatComparisonAnswer.is_benchmark` |
| Sources | `ChatComparisonAnswer.sources` |
| Latency/errors/grounding mode | `ChatComparisonAnswer.latency_ms`, `error_text`, `grounding_mode` |
| Model/config metadata | `ChatComparisonAnswer.variant_metadata` |
| Token/runtime metadata | `ChatComparisonAnswer.runtime_metadata` |
| Feedback selections | `ChatComparisonFeedback.*_label` and answer foreign keys |

## Missing or Still To Confirm

| Variant | Missing details |
|---|---|
| Mistral current chatbot | Exact parameter size; exact context length from Azure Foundry; exact checkpoint/revision beyond `mistral-small-2503`; confirmed quantisation; whether staging pydantic AI/anyllm changes will be used in the experiment or only after it. |
| Qwen3 | Exact checkpoint/revision for `qwen3-30b-a3b-awq`; deployment hardware; confirmed quantisation details beyond AWQ in model id; whether any config changes are planned before launch. |
| EuroLLM | Final `.env.eurollm` model name and public URL should be updated from placeholders; confirm whether experiment uses `utter-project/EuroLLM-9B-Instruct-2512`; deployment hardware; exact quantisation/precision. |
| OpenAI hidden benchmark | Confirm exact currently available model id/version and supported context length; confirm whether `gpt-5.4-mini` is the intended deployed model name. |
| Claude hidden benchmark | Confirm exact currently available model id/version and supported context length; confirm whether `claude-haiku-4-5` is the intended deployed model name. |

## Suggested Mistral Metadata to Add to Environment

The frontend already persists `variant_metadata` for Arena answers. To make future Mistral runs self-describing, add non-secret metadata through environment variables where the Arena frontend is deployed:

| Env var | Suggested value |
|---|---|
| `EUF_CHATBOT_MODEL_NAME` | `azure_ai/mistral-small-2503` |
| `EUF_CHATBOT_MODEL_FAMILY` | `Mistral Small` |
| `EUF_CHATBOT_MODEL_SIZE_HINT` | `24B assumed` |
| `EUF_CHATBOT_PROVIDER` | `Mistral via Azure AI` |
| `EUF_CHATBOT_PROVIDER_RUNTIME` | `Azure AI endpoint via LiteLLM simple-shuffle` |
| `EUF_CHATBOT_NUM_CTX` | `32768` if confirmed; otherwise leave blank |
| `EUF_CHATBOT_TEMPERATURE` | `0.2` |
| `EUF_CHATBOT_RETRIEVAL_CANDIDATE_K` | `3` |
| `EUF_CHATBOT_RETRIEVAL_MIN_SCORE` | `0.4` |
| `EUF_CHATBOT_PROMPT_FAMILY` | `classic_adaptive_rag` |
| `EUF_CHATBOT_PROMPT_VERSION` | `current_production_language_cap` |
| `EUF_CHATBOT_CONFIG_SNAPSHOT_DATE` | Date when the config is frozen for the experiment |

## Practical Recommendation

For the admin dashboard, show a compact per-variant config panel with model id, provider, visible/hidden status, generation settings, retrieval settings, and missing-field warnings. For the experiment report, include the full tables above as an appendix, because the experiment compares complete RAG configurations rather than isolated base models.
