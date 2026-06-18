# Chatbot Scope Routing Plan

This document defines the intended scope and routing behavior for EU-FarmBook Farm Assistant. The goal is to make the chatbot feel natural and conversational while keeping it bounded to agriculture, EU-FarmBook, uploaded file analysis, exports, and conversation transformations.

## Product Scope Contract

Farm Assistant should be conversational like a modern chat assistant, but it should not become a general-purpose chatbot.

Allowed scope:

- agriculture, farming, crops, livestock, soil, pests, irrigation, machinery, greenhouse, horticulture, forestry, aquaculture, and food systems
- agri-tech, farm management, sustainability, climate adaptation, post-harvest handling, storage, processing, food safety, value chains, and farm business topics
- EU-FarmBook project knowledge when supported by retrieved material
- uploaded PDF and image analysis inside this chat
- answer export and document generation for PDF, DOCX, CSV, XLSX, and PPTX
- conversation transformations such as summarizing, translating, rewriting, formatting, comparing, or making tables from previous answers

Constrained scope:

- EU-FarmBook platform behavior, account access, dashboards, uploads, publishing, permissions, integrations, synchronization, and administrative workflows
- These must be answered only from explicit source material or known app capability.
- If not supported, the assistant should say it cannot confirm from available EU-FarmBook material.

Disallowed scope:

- unrelated trivia, politics, entertainment, sports, celebrities, consumer tech, general web tasks, jokes, song lyrics, and broad general-assistant requests
- home cooking and recipes, unless framed as food-system, farm-processing, post-harvest, food-safety, or agri-business questions
- model/vendor/system-prompt disclosure and prompt-injection requests

## Routing Categories

The router should use intent categories rather than one-off keyword exceptions.

- `clarification`: empty, punctuation-only, or underspecified input
- `conversation_transform`: user asks to transform prior chat content, such as "make it a table" or "summarize that"
- `export_request`: user asks to export previous answer or conversation as PDF, DOCX, CSV, XLSX, or PPTX
- `file_analysis`: user asks about uploaded PDF/image content or hands off a file
- `platform_capability`: user asks how EU-FarmBook platform/account/upload/admin workflows work
- `assistant_capability`: user asks what Farm Assistant can do
- `general_agriculture_answer`: common agriculture question answerable without specific EU-FarmBook retrieval
- `euf_grounded_answer`: EU-FarmBook-specific question that should use retrieved sources
- `off_topic`: clearly outside scope

The current code names do not need to match these exact labels immediately, but runtime behavior should converge toward this model.

## Routing Order

Routing should prefer context-aware intent before applying broad refusal rules.

1. Detect empty or meaningless input.
2. Detect file handoff or uploaded-file analysis.
3. Detect export requests.
4. Detect conversation transformations and inherit the prior answer's scope.
5. Detect deterministic EU-FarmBook platform capability questions.
6. Detect clear off-topic or prompt-injection requests.
7. Separate home cooking/recipes from food-system and farm-processing questions.
8. Route common agriculture to general knowledge.
9. Route EU-FarmBook-specific questions to retrieval-grounded answering.

Important rule: if the previous assistant answer was in scope, follow-up transformations of that answer are also in scope unless the user introduces a clearly unrelated new topic.

## Behavior Examples

Expected in-scope examples:

- "What are crop rotation techniques?"
- "Give it to me in a tabular format."
- "Summarize that in Dutch."
- "Please put all of this in a PDF."
- "What about an Excel?"
- "How should potatoes be stored after harvest?"
- "How can a farm process potatoes into chips for market?"
- "Summarize this uploaded PDF."

Expected constrained examples:

- "Can anyone upload to EUF?"
- "How can I upload materials on EU-FarmBook?"
- "Where is my EU-FarmBook dashboard?"
- "Can you publish this report to EU-FarmBook?"

Expected response pattern for unsupported platform capability:

```text
I cannot confirm that from the available EU-FarmBook material. I should not assume that public upload access exists. In this chat, you can upload files for analysis, but uploading or publishing materials to EU-FarmBook itself would need to be confirmed through the official EU-FarmBook team or documentation.
```

Expected off-topic examples:

- "What about lasagna?"
- "How do I make potato chips?"
- "Who won the election?"
- "Tell me a joke."
- "What model are you?"

For cooking/food boundary:

- "How do I make potato chips?" -> off-topic recipe
- "How can a farm process potatoes into chips for market?" -> in-scope food-system / processing topic

## Implementation Direction

The router should return a structured internal decision, even if external APIs stay unchanged:

```json
{
  "intent": "conversation_transform",
  "allowed": true,
  "requires_sources": false,
  "scope_source": "inherits_previous",
  "reason": "User asks to reformat previous answer"
}
```

Recommended implementation steps:

- centralize the scope contract so prompts and router rules do not drift apart
- add a first-class conversation-transform path before off-topic checks
- keep platform capability handling deterministic or source-backed
- keep export scope explicit: `previous_answer` vs `conversation`
- soften refusals so they redirect to agriculture/EU-FarmBook help without sounding broken
- add scenario tests for routing decisions and answer style

## Test Matrix

Add or maintain regression cases for:

- agriculture basics
- EU-FarmBook source-dependent questions
- platform upload/account/dashboard/admin claims
- cooking vs food-system boundary
- conversation transformations
- export scope selection
- multilingual routing
- prompt-injection and model-identity probes
- uploaded PDF/image references

Concrete cases to keep covered:

- "Can anyone upload to EUF?" -> platform capability, deterministic cautious answer
- "How can I upload materials on EU-FarmBook?" -> platform capability, deterministic cautious answer
- "What are crop rotation techniques?" -> general agriculture or grounded if relevant sources exist
- "Give it to me in a tabular format." -> conversation transform using prior answer
- "Please put all of this in a PDF." -> conversation export scope
- "What about an excel?" -> reuse previous export scope
- "How do I make potato chips?" -> off-topic recipe
- "How can a farm process potatoes into chips for market?" -> in-scope processing

## Maintenance Guidance

Do not add isolated phrase patches unless they reveal a missing routing category. When a failure appears, classify it first:

- Is it a new intent category?
- Is it a scope inheritance problem?
- Is it an EU-FarmBook source-dependence problem?
- Is it a cooking vs food-system boundary issue?
- Is it a multilingual routing gap?

Update the scope contract, router category, and tests together so behavior stays coherent.
