# Farm Assistant Changelog

This changelog records Farm Assistant changes from the `master` branch histories. Entries are grouped by commit author date and include the commit hash, branch history, merge marker where applicable, author, and commit subject.

## 2026-06-18

- `115a7a17` [master] - pranavnbapat - feat: improve Farm Assistant scope routing and export handling

## 2026-06-17

- `ebaf3c6e` [master] - pranavnbapat - Organize project documentation under docs

## 2026-06-16

- `2a8d748c` [master] - pranavnbapat - feat(arena): proxy routes for arena variant visibility endpoints
- `3c00a5e7` [master] - pranavnbapat - feat(arena): add opt-in Anthropic LLM provider (LLM_PROVIDER-gated, lazy-imported; Qwen path unchanged) + 1x5 plan/deploy docs
- `f3ad5297` [master] - pranavnbapat - feat: proxy comparisons/benchmark endpoint to chat backend
- `66102006` [master] - pranavnbapat - feat: emit real vLLM token usage in chat stream timing event

## 2026-06-10

- `5915fbf5` [master] - pranavnbapat - feat: add branded headers, footers, and source citations to PDF exports

## 2026-06-09

- `48e81636` [master] - pranavnbapat - feat: add multilingual document export intent classifier
- `b1175ae8` [master] - pranavnbapat - feat: add explicit document download buttons and concise export responses
- `ea998cf4` [master] - pranavnbapat - feat: generate assistant responses as PDF, DOCX, CSV, XLSX, and PPTX

## 2026-06-08

- `669ec28b` [master] - pranavnbapat - fix: improve assistant grounding, language handling, and memory consolidation
- `50ebd83b` [master] - pranavnbapat - fix: ground follow-ups, memory, citations, and response language

## 2026-05-28

- `474da21b` [master] - pranavnbapat - fix(prompt): anchor answer language on the question, not the retrieved sources

## 2026-05-22

- `80bfbacd` [master] - pranavnbapat - feat(chatbot-api): proxy experiment and comparison endpoints for evaluation flows

## 2026-05-11

- `afca2d03` [master] - pranavnbapat - application improved
- `4f83b365` [master] - pranavnbapat - fix(farm-assistant): tighten long-term memory filtering and reuse
- `97a723dd` [master] - pranavnbapat - fix(farm-assistant): harden routing guards and emit canonical source URLs
- `ebb2ca44` [master] - pranavnbapat - feat(farm-assistant): add LLM retrieval integration and tighten off-topic routing

## 2026-05-08

- `32048950` [master] - pranavnbapat - application improved

## 2026-05-07

- `4710c71e` [master] - pranavnbapat - application improved
- `7c9279f0` [master] - pranavnbapat - Application enhanced
- `19a71d3b` [master] - pranavnbapat - fix(dev): stop using incorrect backend-admin basic auth credentials for chat proxying

## 2026-04-28

- `1b6067b6` [master] - pranavnbapat - fix: add upstream backend-admin auth support and simplify run.sh backend selection

## 2026-04-04

- `6c5bc2de` [master] - pranavnbapat - sources showed properly

## 2026-04-03

- `d4b4dffb` [master] - pranavnbapat - Layout improved
- `10d62c00` [master] - pranavnbapat - refine composer UX, voice controls, and frontend polish
- `705d07f4` [master] - pranavnbapat - align README, architecture, and data inventory with current codebase

## 2026-04-02

- `5c3ef43e` [master] - pranavnbapat - remove Redis caching and related deployment/config code
- `38946326` [master] - pranavnbapat - add Redis deployment support and align cache configuration
- `cc8b6f8a` [master] - pranavnbapat - building and pushing image automated
- `e5c06d57` [master] - pranavnbapat - improve chat UX and simplify dependency management
- `6c550bd7` [master] - pranavnbapat - application improved
- `286bd8a2` [master] - pranavnbapat - expose chatbot APIs and align chat flow, config, and history handling

## 2026-02-25

- `67bbdb2c` [master] - pranavnbapat - context limit increased
- `e779a501` [master] - pranavnbapat - file attachment support added
- `66ad9653` [master] - pranavnbapat - user profiling added, polished UI

## 2026-02-24

- `46452f18` [master] - pranavnbapat - commit before major changes

## 2026-02-21

- `ea5416f8` [master] - pranavnbapat - working and improved version till here
- `16376079` [master] - pranavnbapat - working version till here
- `742e735b` [master] - pranavnbapat - code updated

## 2025-12-03

- `7c57a8ab` [master] - pranavnbapat - farm assistant cleaned, wired with backend admin

## 2025-12-02

- `68049fc7` [master] - pranavnbapat - code cleaned
- `500d9df3` [master] - pranavnbapat - speaker icon added
- `9e50e494` [master] - pranavnbapat - voice options removed, and made it streamlined.
- `73022b0d` [master] - pranavnbapat - focus to input box added
- `9d60fb95` [master] - pranavnbapat - farm assistant added for CORS and CSRF

## 2025-11-28

- `7388cc1a` [master] - pranavnbapat - onnx files support added to docker
- `fc236bb4` [master] - pranavnbapat - onnx files ignored
- `b6ef0610` [master] - pranavnbapat - Stop tracking ONNX model files
- `20bc980e` [master] - pranavnbapat - farm assistant improved with voice and nice login screen

## 2025-10-20

- `bfc04a1d` [master] - pranavnbapat - parallel queues added

## 2025-10-19

- `30ad0202` [master] - pranavnbapat - intent recognition added

## 2025-10-17

- `1c614504` [master] - pranavnbapat - some models disabled

## 2025-10-16

- `cf94fe0d` [master] - pranavnbapat - intent detection reverted. Code reverted.
- `78e707dd` [master] - pranavnbapat - intent recognition added, needs a lot of tuning.

## 2025-10-15

- `29068a0f` [master] - pranavnbapat - new deepseek models added
- `4c652ba2` [master] - pranavnbapat - timer added

## 2025-10-14

- `75485829` [master] - pranavnbapat - 2 new models added.

## 2025-10-13

- `b6dc43b6` [master] - pranavnbapat - citation and prompt improved
- `9b4ec1dc` [master] - pranavnbapat - docs blocked
- `e09d5ba2` [master] - pranavnbapat - application segregated, streaming added
- `0a394c5e` [master] - pranavnbapat - Initial commit

## 2026-06-22

- `350d11a0` [master] - pranavnbapat - fix(anthropic): drop top_p so Claude isn't sent both temperature and top_p

## 2026-06-19

- `5a795d4a` [master] - pranavnbapat - feat: add per-backend generation queues and changelog tooling

## 2026-06-23

- `43cd3ad5` [master] - pranavnbapat - fix(markdown): strip orphan bullet/dot lines incl. nbsp/zero-width from model output
