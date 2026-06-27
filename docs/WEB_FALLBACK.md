# Farm Assistant — Web-Search Fallback: Architecture & Flow

Plain-language explanation of how Farm Assistant decides between **EU-FarmBook
knowledge objects (KOs)**, **the web**, and **the model's own knowledge**, what the
exact thresholds are, the caveats, and how to make it more robust.

---

## 1. The big picture (one turn, end to end)

```
                          ┌──────────────────────┐
   User asks a question → │ Router (LLM + rules) │  decides the "turn mode"
                          └──────────┬───────────┘
                                     │
   ┌─────────────┬─────────────┬─────┴───────┬───────────────┬──────────────┐
   ▼             ▼             ▼             ▼               ▼              ▼
off_topic   conversation  history_only  general_       platform_       NORMAL
(refuse)    _only         (recap)       knowledge      operation     (retrieval)
   │             │             │             │               │              │
   │             │             │   answer from model's       │              │
   │             │             │   own knowledge,            │              │
   │             │             │   NO sources, NO web        │              │
   └─────────────┴─────────────┴─────────────┴───────────────┘              │
                                                                            ▼
                                                        ┌───────────────────────────────┐
                                                        │  RETRIEVAL + WEB DECISION     │
                                                        │  (only this mode can use web) │
                                                        └───────────────────────────────┘
```

**Key point:** the web fallback only ever runs on the **NORMAL** (retrieval) turn.
A question the router sends to `general_knowledge` is answered from the model's own
(uncited) knowledge and **never** reaches the web step — even if web search is on.

---

## 2. The NORMAL turn in detail (where the web decision lives)

```
 [NORMAL turn]
      │
      ▼
 (1) Ask OpenSearch  ──►  POST /llm_retrieve
     k = max(top_k, RETRIEVAL_CANDIDATE_K=10)
      │
      ▼
     items[]  (each has a raw _score from a hybrid neural+keyword search)
      │
      ▼
 (2) JUNK FILTER ─ drop any item whose _score < RETRIEVAL_MIN_SCORE (=1.0)
      │
      ▼
 (3) Build contexts from survivors (keep best TOP_K=5, collapse chunks per KO,
     cap total at MAX_CONTEXT_CHARS=24000)
      │
      ▼
 (4) RELEVANCE ESTIMATE
     quality = estimate_retrieval_quality(query, top 3 items)   →  a 0..1 number
     (NOT the OpenSearch score — see §3)
      │
      ├── quality < 0.15  ──►  throw away ALL KO contexts (treat as "found nothing")
      │
      ▼
 (5) WEB FALLBACK DECISION   (only if WEB_FALLBACK_ENABLED = true)
      │
      │   needs_web =  (no KO contexts left)            ← empty
      │             OR (quality < 0.35)                 ← weak  (WEB_FALLBACK_QUALITY_THRESHOLD)
      │
      ├── needs_web = false ─────────────►  use KO context only        → "Grounded in EU-FarmBook"
      │
      └── needs_web = true  ─► run the WEB CHAIN (§4), then:
             • KO had some context + web added  →  "Grounded in EU-FarmBook + external sources"
             • only web produced context        →  "Grounded in external sources"
             • nothing anywhere                 →  "General agricultural guidance" (model knowledge)
      │
      ▼
 (6) Send KO + web sources to the LLM. It writes the answer and cites [1], [2]...
      │
      ▼
 (7) Only the sources the model actually cited are shown as clickable links in the UI.
```

### The three numbers, and what each one really does

| # | Setting | Default | Job |
|---|---------|---------|-----|
| 1 | `RETRIEVAL_MIN_SCORE` | `1.0` | Junk floor on the **raw OpenSearch score**. Removes weak hits before we build context. |
| 2 | `RETRIEVAL_DROP_THRESHOLD` | `0.15` | If the lexical relevance estimate is this low, drop **all** KO context → behave as "found nothing". |
| 3 | `WEB_FALLBACK_QUALITY_THRESHOLD` | `0.35` | The "weak" trigger. Below this, fire the web fallback (and merge web with any KO context). |

Also: `WEB_FALLBACK_MAX_RESULTS=4`, `WEB_FALLBACK_MAX_CHARS=6000`, `WEB_FETCH_TIMEOUT=6.0`.

---

## 3. What "quality" actually measures (important)

`quality` is **not** the OpenSearch `_score`. It is a separate, simple
**word-overlap** check (`estimate_retrieval_quality` in `context_service.py`):

1. Take the user's (normalized) query.
2. Take the **title + subtitle + description** of the top 3 surviving items.
3. Count how many words overlap: `overlap = |query_words ∩ item_words| / min(...)`.
4. Average over the 3 items → a number from 0 to 1.

**Why not use the OpenSearch score directly?** Because `/llm_retrieve` returns a
*hybrid* (neural + keyword) score that is **not calibrated** — "4.0" means different
things for different queries, so a fixed cutoff would misfire. The word-overlap
estimate is a stable, query-relative "are these results actually about the question?"
signal. That is the deliberate design choice.

### Optional: semantic mode (`RELEVANCE_MODE=semantic`)

Word overlap is lexical and misses meaning. When `RELEVANCE_MODE=semantic`, farm_assistant
asks scout (`include_semantic_score=true`) for a per-chunk **`semantic_score`** — a
calibrated, query-independent cosine from a plain `content_embedding` neural query (no
hybrid normalization). The gate thresholds the **best** (max) `semantic_score` among the
top-3 chunks using `SEMANTIC_DROP_THRESHOLD` (0.88) / `SEMANTIC_WEB_THRESHOLD` (0.90)
instead of the overlap thresholds. If items carry no `semantic_score`, it falls back to
word-overlap automatically. Default stays `overlap` (no behavior change).

Thresholds calibrated on msmarco over 9 probes: on-topic best-chunk scores clustered
**>=0.91**, off-topic **<=0.87** (scale is (1+cos)/2). The margin is narrow (~0.04), so
this is msmarco-specific and somewhat sensitive — re-calibrate per model (e.g. minilm) and
from ClickHouse `/llm_retrieve` logs.

---

## 4. The web search chain (how a web answer is built)

```
 web_search_and_build_contexts(query)
      │
      ▼
 _chain_search  ── try providers in order: WEB_SEARCH_PROVIDERS
      │           default: tavily → staan → brave → duckduckgo → wikipedia
      │
      │   for each provider:
      │     • no API key set?           → skip, go to next
      │     • provider errors / quota?  → log, go to next
      │     • returns 0 allowlisted hits→ go to next
      │     • returns hits              → STOP, use these
      │
      ▼
 keep only results whose domain is in WEB_TRUSTED_DOMAINS   (allowlist, enforced here)
      │
      ▼
 dedupe by domain, cap to MAX_RESULTS (4)
      │
      ▼
 for each result:
     • Tavily/Wikipedia already return clean text  → use it directly
     • Brave/DuckDuckGo return links only          → fetch the page + extract main
                                                      text with trafilatura
      │
      ▼
 build [S?]-headed context blocks + Source links (title → URL), within char budget
      │
      ▼
 hand back to the NORMAL turn (step 6 above)
```

**Providers today:** with only a Tavily key set, the live chain is effectively
**Tavily → DuckDuckGo → Wikipedia** (staan/brave skipped until their keys exist).
All search/fetch happens in the backend — the model never browses by itself.

---

## 5. Caveats (where this can be wrong or weak)

1. **Relevance is judged by word overlap, not meaning.** A perfectly relevant KO
   that uses *different words* than the question (synonyms, another language) can
   score low and wrongly trigger the web. Conversely, a KO sharing words but not
   meaning can score high and block the web.
2. **English-biased.** The overlap check tokenizes English-style words (≥3 letters).
   It is weaker for non-English queries.
3. **Only metadata is checked.** Quality looks at title/subtitle/description of the
   top 3 items — not the actual chunk body that will ground the answer.
4. **Raw retrieval score is barely used.** It is only a junk floor (`1.0`); it does
   not influence the web decision, because hybrid scores aren't comparable.
5. All three thresholds are now env-tunable (`RETRIEVAL_MIN_SCORE`,
   `RETRIEVAL_DROP_THRESHOLD`, `WEB_FALLBACK_QUALITY_THRESHOLD`).
6. **Foundational questions may skip web entirely.** If the router labels a basic
   question `general_knowledge`, it answers from the model's own knowledge and never
   reaches retrieval or web — so the web fallback can't help there.
7. **Latency.** Web search = a search API call + fetching up to 4 pages + extraction,
   all before the answer starts streaming. A "Searching trusted external sources…"
   status is shown, but it adds seconds.
8. **Empty-result fall-through is greedy.** If a provider returns nothing, the chain
   tries the next one — so a genuinely no-result query can hit all providers (extra
   latency/cost).
9. **Extraction quality varies.** `trafilatura` works on HTML; PDF links and unusual
   pages may yield little or no text.
10. **Sources only show if cited.** If the model doesn't write `[1]`, that source
    won't appear in the UI, even though it was used as context.
11. **Allowlist is all-or-nothing.** Clearing `WEB_TRUSTED_DOMAINS` blocks every web
    result (safe default), but there's no per-tier trust weighting.

---

## 6. How to make it more robust (roadmap, roughly ordered by value)

1. **Decide on meaning, not words.** Replace the word-overlap estimate with a
   semantic signal: either embedding cosine-similarity between the query and the
   retrieved chunk text, or a **cross-encoder rerank score**. `beacon` already runs a
   cheap rerank — reuse that. This fixes caveats #1–#4 in one move and gives a
   calibrated, language-robust number to threshold.
2. **Tune the thresholds with data, not guesses.** Every search is already logged to
   ClickHouse. Look at real distributions of quality vs. answer usefulness and set
   `0.15` / `0.35` from evidence. Promote the hardcoded `0.15` to a setting.
3. **Cache web results in Valkey** (keyed by normalized query + provider). Cuts
   latency and cost, and makes answers stable instead of changing as the web changes.
4. **Add a global web deadline + per-provider circuit breaker.** Bound total web time
   so a slow page can't stall the stream; skip a provider that's recently failed.
5. **Rerank KO + web together** before picking the final top-k, so the best sources
   win the limited context budget regardless of origin (instead of "KO first, then
   web appended").
6. **Let `general_knowledge` optionally use the web** for foundational gaps, so basic
   questions can also be grounded + cited instead of answered from memory (caveat #6).
   This is also the partner's original "bring in broader knowledge" request.
7. **Use the provider's own relevance score** (e.g. Tavily returns scores) as an extra
   signal, and **de-duplicate** web pages that are really the same document as a KO.
8. **Record observability**: which provider served, the quality value, KO-vs-web usage
   per turn → ClickHouse, so the thresholds and provider order can be tuned over time.
9. **Trust tiers in the allowlist** (e.g. gov/edu > journal > general) to weight or
   order sources, instead of a flat allow/deny.

> Governance note: turning this on (`WEB_FALLBACK_ENABLED=true`) means the assistant
> sources from **outside** EU-FarmBook — the thing the Management Board has so far
> declined. Confirm policy before enabling in production.
