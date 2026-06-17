# Farm Assistant Arena Experiment Plan

## 1. Purpose

This document explains how to compare several versions of the EU-FarmBook chatbot in a fair, manageable, and statistically useful way.

The main goal is:

> Select the best complete chatbot configuration for possible production use.

A complete configuration is a combination of:

- one language model;
- one RAG method;
- one fixed set of prompts and generation settings;
- one fixed EU-FarmBook knowledge-base snapshot.

The experiment should measure answer quality, but production selection must also consider grounding, safety, reliability, response time, and cost.

## 2. Proposed Configurations

### 2.1 Model families

Use five model families, grouped by licence:

Open-weight (self-hosted via vLLM):

1. Qwen
2. Mistral
3. EuroLLM

Proprietary (hosted API):

4. OpenAI/ChatGPT
5. Anthropic/Claude

There are two proprietary model families, OpenAI and Anthropic, not one. Both are run through the same `farm_assistant` RAG pipeline as the open-weight families — only the final generation call targets the provider API, so retrieval, prompts, citation rules, and output limits stay equivalent across all families. In the arena they appear as blind cards, indistinguishable from every other configuration.

The exact model identifiers must be agreed and frozen before the experiment. For example, writing only "GPT" or "Qwen" is not sufficient because providers can release or update models during the study.

For every model, record:

- exact provider and model identifier;
- API or deployment version;
- hosting location;
- context-window limit;
- temperature, `top_p`, and output-token limit;
- cost or local hardware requirement;
- date on which the configuration was frozen.

### 2.2 RAG methods

The intended methods are:

#### Method 1: Standard routed RAG

The system:

1. decides whether the question requires retrieval;
2. prepares the search query;
3. retrieves EU-FarmBook documents;
4. generates one answer from the retrieved evidence.

This is the baseline. It does not retry retrieval after poor results and does not verify the completed answer.

The current `farm_assistant` service is the closest existing implementation.

#### Method 2: Corrective RAG with reflection

The system:

1. routes the question;
2. retrieves documents;
3. grades the retrieval quality;
4. rewrites and retries the search when retrieval is weak;
5. generates an answer;
6. checks whether important claims are supported by the retrieved evidence.

The current `agentic_farm_assistant` already implements the corrective retrieval loop. However, answer-level reflection must be confirmed as complete before this method is officially called "self-reflection RAG."

Retrieval correction and answer reflection are different:

- Retrieval correction checks whether the search results are useful.
- Answer reflection checks whether the final answer is supported by those results.

#### Method 3: Metadata-aware corrective RAG with reflection

This method follows Method 2 but uses an advanced retriever that can filter, rank, or route with EU-FarmBook structured metadata.

Possible metadata includes:

- country or geographical coverage;
- language;
- agricultural sector or topic;
- project;
- content or knowledge-object type;
- publication date;
- target audience;
- practice, result, or output category.

The exact metadata fields and ranking rules must be agreed before implementation. This method must produce meaningfully different retrieval behaviour. It should not be only a new label for the existing retriever.

### 2.3 The configuration set (1 × 5)

The current experiment fixes a **single RAG method** — Standard routed RAG (Method 1), i.e. the `farm_assistant` pipeline — and varies **only the model family**. That gives five configurations:

| RAG method | Qwen3 | Mistral | EuroLLM | OpenAI | Anthropic |
|---|---:|---:|---:|---:|---:|
| Standard routed RAG | Configuration 1 | Configuration 2 | Configuration 3 | Configuration 4 | Configuration 5 |

- Configuration 1 — `farm_assistant` + Qwen3 (self-hosted vLLM)
- Configuration 2 — Mistral (existing external deployment)
- Configuration 3 — `farm_assistant` + EuroLLM (self-hosted vLLM)
- Configuration 4 — `farm_assistant` + OpenAI API
- Configuration 5 — `farm_assistant` + Anthropic API

Methods 2 (corrective RAG) and 3 (metadata-aware corrective RAG), described above, are out of scope for this run and kept only as future options. All five configurations must use the same knowledge-base snapshot and, as far as possible, equivalent prompts, citation rules, output limits, and safety instructions.

## 3. Recommended Experiment Structure

Use two evaluation stages.

### Stage 1: Automated screening

Run all nine configurations against the same curated question bank before asking agricultural experts to evaluate them.

Use approximately 30 to 50 questions for initial screening. The final bank may be larger if it can be prepared and reviewed reliably.

Measure:

- whether the correct evidence was retrieved;
- factual correctness;
- citation validity and claim support;
- relevance;
- handling of missing or conflicting evidence;
- safety and appropriate restraint;
- response latency;
- timeout and error rate;
- input and output token usage;
- estimated API or infrastructure cost.

Remove configurations that fail agreed minimum requirements. Experts should not spend time evaluating configurations that are already technically unsuitable.

The preferred result is a shortlist of approximately four strong configurations for human evaluation. Keep all nine in the human study only if estimating model effects, RAG-method effects, and their interaction is a formal research requirement.

### Stage 2: Human Arena evaluation

Show two anonymised answers at a time.

For each comparison:

1. Show one agricultural question.
2. Show Answer A and Answer B.
3. Hide model and method identities.
4. Randomize which configuration appears on the left and right.
5. Ask the evaluator to select:
   - Answer A;
   - Answer B;
   - Tie;
   - Neither answer is acceptable.
6. Collect only a small number of secondary ratings.
7. Save the answer texts, sources, configuration snapshots, assignment data, and feedback.

Do not use a winner-stays or king-of-the-hill tournament for the formal study.

In a winner-stays design, the winner of A versus B is compared with C, then the next winner is compared with D. This creates problems:

- later comparisons depend on earlier results;
- configurations receive unequal opponents;
- the first configuration can appear repeatedly;
- comparison order can influence the final ranking;
- simple win totals become difficult to interpret fairly.

Instead, use a balanced comparison schedule. Across the whole study, each configuration should receive similar exposure to:

- every other relevant configuration;
- question categories;
- left and right positions;
- early and late session positions.

## 4. Participant Workload

The target session duration is 15 to 20 minutes.

Recommended workload:

- 8 to 12 pairwise comparisons per participant;
- short instructions and one practice comparison;
- optional comment at the end;
- progress indicator;
- ability to stop without losing completed responses.

Thirty-two comparisons in one sitting may cause fatigue, rushed judgments, and participant dropout. If more judgments per expert are essential, use two shorter sessions instead of one long session.

Thirty participants can be useful for:

- a pilot study;
- estimating completion time and tie rate;
- identifying interface or question problems;
- comparing a small number of shortlisted finalists.

Thirty participants do not automatically guarantee statistically significant results for nine configurations.

For example, 30 participants completing 32 comparisons produce 960 recorded decisions, but these are not 960 independent observations. Decisions from the same person are related, and answers to the same question are related. The statistical analysis must account for this clustering.

A reasonable initial planning target for a larger study is at least 60 participants completing approximately 12 comparisons each. The final sample size should be calculated through simulation after pilot data provide realistic estimates of:

- expected win-rate differences;
- tie and neither rates;
- disagreement between participants;
- variation between questions;
- dropout and incomplete-session rates.

## 5. Question Bank

Use a curated and reviewed question bank for the formal experiment. Do not use only participant-written questions because every configuration then receives a different and potentially incomparable test.

The question bank should include:

- simple factual retrieval;
- practical agricultural advice;
- multi-document synthesis;
- questions for which structured metadata should help;
- ambiguous or incomplete questions;
- questions with little or no supporting evidence;
- questions containing incorrect assumptions;
- conflicting or uncertain evidence;
- safety-sensitive agricultural advice;
- multilingual questions;
- off-topic and chatbot-capability questions.

For every question, record:

- stable question ID;
- exact wording;
- language;
- agricultural domain;
- question type and difficulty;
- expected relevant documents, where available;
- whether the knowledge base contains enough evidence;
- expected safe behaviour;
- reviewer names and review status.

Questions should be reviewed by agricultural experts and by someone familiar with the EU-FarmBook data.

Participants may optionally submit their own questions in a separate exploratory section. Those results must not be mixed directly with the controlled question-bank results.

## 6. Evaluation Criteria

### 6.1 Primary outcome

Use one clear primary outcome:

> Which answer is better overall for this agricultural question?

Allowed responses:

- Answer A;
- Answer B;
- Tie;
- Neither is acceptable.

This primary result is used for the main ranking.

### 6.2 Secondary outcomes

Keep the secondary questionnaire short. Recommended criteria are:

- factual and evidential trustworthiness;
- relevance;
- practical usefulness;
- clarity;
- handling of uncertainty, when the question contains uncertainty or insufficient evidence.

Do not require six repeated winner selections after every pair unless the research team can justify the added burden.

Do not create an informal "cocktail metric" after collecting data. If several metrics are to be combined, define and document the weights before the study begins.

### 6.3 Operational measures

Store these measures automatically:

- response time;
- timeout or backend error;
- returned sources;
- grounding mode;
- token counts, where available;
- estimated cost;
- retrieval iterations;
- retrieval grade;
- reflection or verification result;
- model, prompt, retriever, code, and corpus versions.

Do not show response time or model identity while participants judge answer quality. These details may bias preferences.

## 7. System Architecture

### 7.1 High-level flow

```text
EU-FarmBook Arena UI
        |
        | requests the participant's next assigned task
        v
Experiment API / Orchestrator
        |
        |-- loads the fixed experiment protocol
        |-- assigns a curated question
        |-- assigns two configurations
        |-- randomizes left/right display
        |-- starts or retrieves generated answers
        v
Variant Adapter Layer
        |
        |-- Standard routed RAG
        |-- Corrective RAG with reflection
        |-- Metadata-aware corrective RAG with reflection
        |
        | each method can use:
        |-- OpenAI
        |-- Mistral
        |-- Qwen
        v
Shared frozen EU-FarmBook knowledge base
        |
        v
Django experiment storage and result export
```

### 7.2 Frontend responsibilities

`eu-farmbook-frontend` should:

- show participant information and instructions;
- obtain the next assigned comparison;
- show exactly two anonymised answers;
- randomize presentation only according to the backend assignment;
- support A, B, Tie, and Neither choices;
- collect the selected secondary ratings;
- show progress and completion;
- prevent accidental duplicate submission;
- avoid exposing hidden configuration identities.

The frontend should not decide which pair is statistically needed. Pair and question assignment should be controlled by the experiment backend so that assignments remain balanced across users and devices.

### 7.3 Experiment backend responsibilities

`django_euf_admin` should be the main experiment record and source of truth.

It should store:

- experiment protocol and version;
- participant and study session;
- assigned tasks;
- question ID and question version;
- the two configuration IDs;
- random left/right assignment;
- answer texts and source snapshots;
- model and RAG configuration snapshots;
- primary decision and secondary ratings;
- task start, answer-ready, and submission times;
- incomplete, skipped, or invalid task state;
- optional comment;
- exclusion or data-quality flags.

The current comparison tables already preserve runs, answers, variants, metadata, and feedback. They will need to be extended or complemented to represent a complete study protocol, assigned pairwise tasks, ties, neither choices, and question-bank records.

### 7.4 Chatbot responsibilities

`farm_assistant` should provide the standard routed RAG treatment.

`agentic_farm_assistant` should provide the corrective treatment and, once complete, answer-level verification.

The metadata-aware method is implemented in `agentic_farm_assistant` as the fixed `metadata_corrective` retrieval profile. It shares the controller with the `corrective` profile and adds conservative metadata extraction plus filtered `/llm_retrieve` requests. It must continue to satisfy these conditions:

- it has an explicit configuration ID;
- its metadata behaviour is testable;
- it does not change unrelated prompts or generation behaviour;
- every run records the metadata filters and ranking decisions used.

The chatbot services should return a common response shape containing:

- answer text;
- sources;
- grounding mode;
- response latency;
- error details;
- model identifier;
- method identifier;
- runtime and retrieval telemetry.

### 7.5 Supporting services

`api-core`, `api-database`, or `django_euf_admin` may require changes if the metadata-aware retriever needs new search filters or structured fields.

Do not add these changes until the third RAG method and required metadata fields have been agreed.

## 8. Configuration Registry

Every configuration must have a stable identifier, for example:

```text
standard_openai_v1
standard_mistral_v1
standard_qwen_v1
corrective_openai_v1
corrective_mistral_v1
corrective_qwen_v1
metadata_corrective_openai_v1
metadata_corrective_mistral_v1
metadata_corrective_qwen_v1
```

Each registry entry should include:

- model provider and exact model ID;
- RAG method and method version;
- backend endpoint;
- prompt version;
- retriever version;
- knowledge-base snapshot;
- embedding model and index version;
- metadata fields used;
- generation parameters;
- code repository and commit;
- enabled status;
- experiment eligibility;
- freeze date.

Changing any important setting after the study begins requires a new configuration ID. Otherwise, results from different systems could be combined incorrectly.

## 9. Pair Assignment and Randomization

The backend should generate a balanced assignment plan before or during participant enrolment.

The plan should:

- give each configuration similar total exposure;
- give important configuration pairs similar exposure;
- distribute question categories across configurations;
- balance left and right display;
- rotate task order;
- avoid showing the same question repeatedly to one participant;
- avoid showing the same answer twice to the same participant;
- avoid assigning tasks based on previous winners.

Randomization must be reproducible. Store the randomization seed or the resulting assignment record.

The initial interface can stream answers live, but pre-generating answers is methodologically cleaner when possible because it:

- prevents one participant from receiving a different answer due only to model randomness;
- reduces waiting and dropout;
- makes answer auditing possible before display;
- lowers the risk of backend failures during expert sessions.

If answers are generated live, use controlled generation settings and store every exact output. Never regenerate an answer after the participant has started judging it.

## 10. Statistical Analysis

Do not rank systems using only raw win percentages.

Use a paired-comparison model such as Bradley-Terry, extended where necessary to support ties. The analysis should account for:

- configuration identity;
- participant;
- question;
- left or right position;
- question category;
- language;
- comparison order;
- model family;
- RAG method;
- model-by-method interaction.

Participant and question should be treated as repeated or clustered sources of variation.

Report:

- estimated probability that one configuration beats another;
- ranking with 95% confidence intervals;
- uncertainty in the rank;
- tie and neither rates;
- results by agricultural domain and language;
- model-family effect;
- RAG-method effect;
- model-by-method interaction;
- latency, cost, errors, safety, and grounding alongside preference.

Use bootstrap or model-based confidence intervals that preserve participant and question clustering.

Pre-register:

- primary hypothesis;
- primary outcome;
- minimum sample size;
- exclusion rules;
- stopping rule;
- tie handling;
- missing-data handling;
- metric weights, if any;
- primary and exploratory analyses.

## 11. Production Selection Rules

The most preferred chatbot is not automatically the production winner.

A configuration must first pass minimum gates for:

- citation and grounding quality;
- safety;
- factual correctness;
- acceptable failure and timeout rate;
- acceptable response time;
- sustainable cost and infrastructure;
- privacy and provider compliance.

Among configurations that pass these gates, human preference can be used as the main selection criterion.

The final decision report should explain any trade-off. For example, a slightly less preferred model may be selected if it is substantially safer, faster, cheaper, or easier to operate.

## 12. Work Required Before Implementation

### Decisions required from the research team

Ask the other team to confirm:

1. The exact OpenAI, Mistral, and Qwen model versions.
2. The exact definition of each RAG method.
3. Whether self-reflection includes retrieval correction, answer verification, or both.
4. The EU-FarmBook metadata fields for the advanced retriever.
5. Whether the primary goal is deployment selection or estimation of model and method effects.
6. Minimum quality, safety, latency, reliability, and cost thresholds.
7. Required agricultural domains and languages.
8. Expected participant count and participant expertise.
9. Whether the HHAI 2026 questionnaire, protocol, analysis code, or anonymised data structure can be shared.
10. Agreement to conduct a pilot before fixing the final sample size.

### Work for the EU-FarmBook team

1. Freeze and document the current knowledge-base snapshot.
2. Create and review the curated question bank.
3. Run automated baselines for the current `farm_assistant` and `agentic_farm_assistant`.
4. Complete answer-level reflection if it is part of Method 2.
5. Define and implement the metadata-aware retriever for Method 3.
6. Add exact model adapters and versioned configuration records.
7. Pilot the interface and assignment logic internally.
8. Run a small ambassador pilot.
9. Use pilot data for workload and sample-size simulation.
10. Freeze the full protocol before the main study.

## 13. Suggested Delivery Phases

### Phase 1: Protocol agreement

Deliver:

- approved research question;
- approved nine-cell configuration matrix;
- exact method definitions;
- initial question taxonomy;
- evaluation criteria;
- production eligibility thresholds.

### Phase 2: Technical baseline

Deliver:

- frozen corpus and index;
- configuration registry;
- automated run harness;
- quality, latency, failure, and cost report for all available configurations.

### Phase 3: Missing treatments

Deliver:

- answer-level reflection;
- metadata-aware retrieval;
- common telemetry and response format;
- automated tests proving that the three methods behave differently as specified.

### Phase 4: Arena study workflow

Deliver:

- question-bank storage;
- balanced pair assignment;
- pairwise Arena interface;
- tie and neither choices;
- progress and completion flow;
- study-session persistence;
- blinded result export.

### Phase 5: Pilot

Deliver:

- internal usability results;
- completion-time distribution;
- dropout and missing-data rates;
- tie and neither rates;
- initial variance and effect estimates;
- corrected full-study sample-size recommendation.

### Phase 6: Main study and analysis

Deliver:

- frozen experiment release;
- participant study;
- quality-controlled dataset;
- statistical analysis;
- production recommendation with quality, safety, latency, and cost trade-offs.

## 14. Main Risks

### Too many comparisons

Long sessions increase fatigue and dropout. Keep the main session within 15 to 20 minutes.

### Treatments are not genuinely different

Calling methods "standard," "self-reflection," and "advanced" is not enough. Each treatment must have a precise and testable behaviour.

### Model updates during the experiment

Provider aliases can change. Use fixed versions where available and record every runtime model identifier.

### Unbalanced pair exposure

Pure random selection can produce too many observations for some pairs and too few for others. Use a balanced assignment algorithm.

### Question leakage or memorisation

Keep the formal question bank controlled. Separate development questions from final evaluation questions.

### Evaluator bias

Hide model identities, randomize left/right placement, standardize formatting, and avoid showing latency before judgment.

### Preference without factual quality

Fluent answers can be preferred even when unsupported. Use automated and expert grounding checks as production gates.

### Incomplete methods

The current repository does not yet contain three fully independent RAG treatments. Do not start the full 3 x 3 human experiment until all treatments are complete and validated.

## 15. Current Repository Position

The current Arena already provides:

- configurable backend variants;
- anonymised answer labels;
- randomized answer order;
- simultaneous answer generation;
- source and latency display;
- structured comparison feedback;
- variant and runtime metadata snapshots;
- Django persistence and basic reporting.

The current Arena does not yet provide:

- a fixed question bank;
- participant task assignments;
- balanced pair scheduling;
- a pairwise-only study workflow;
- Tie or Neither choices;
- experiment progress and completion state;
- a formal protocol version;
- explicit repeated-measures analysis support;
- three completed and validated RAG treatments.

The existing implementation is therefore a useful technical foundation, but it should be extended into a controlled experiment system before the proposed research study begins.

## 16. Research Basis

The design follows established principles from paired-comparison experiments and human evaluation:

- Pairwise judgments are generally easier than ranking many answers simultaneously.
- Bradley-Terry-type models estimate relative strength from paired outcomes.
- Balanced incomplete-block designs reduce participant workload while preserving comparisons across treatments.
- Repeated judgments from the same participant and repeated use of the same questions are correlated and should not be treated as independent.
- Position and order effects must be randomized and included in analysis.
- Long questionnaires can reduce completion and response quality.

Useful starting references:

1. Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference," 2024.  
   <https://arxiv.org/abs/2403.04132>
2. Bradley and Terry, "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons," 1952.  
   <https://doi.org/10.2307/2334029>
3. Galesic and Bosnjak, "Effects of Questionnaire Length on Participation and Indicators of Response Quality in a Web Survey," 2009.  
   <https://doi.org/10.1093/poq/nfp031>

## 17. Recommended Final Setup

In simple terms:

1. Build nine clearly defined model-and-method configurations.
2. Test all nine automatically on the same reviewed questions.
3. Remove unsafe, unreliable, poorly grounded, or impractical systems.
4. Ask experts to compare two anonymous answers at a time.
5. Give each expert 8 to 12 balanced comparisons.
6. Let experts choose A, B, Tie, or Neither.
7. Analyse the data while accounting for participant and question differences.
8. Select only from configurations that pass technical production gates.

This approach is more reliable and less tiring than showing all systems together or using a winner-stays tournament.
