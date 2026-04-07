# Graded Reader Generator

An AI pipeline that generates difficulty-controlled serialised fiction for English language learners, using multi-agent LLM orchestration and ML classifiers. Originally built to target CEFR levels, the project has evolved toward vocabulary-frequency-band control as a more empirically grounded approach to text difficulty --- with CEFR retained as the learner-facing label.

## Table of Contents

- [Motivation](#motivation)
- [How It Works](#how-it-works)
- [Difficulty Control](#difficulty-control)
- [The Classifier](#the-classifier)
- [Development Process](#development-process)
- [Key Findings](#key-findings)
- [Example Output](#example-output)
- [Research Background](#research-background)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Motivation

Graded readers --- texts written to be understood by learners at a specific language level --- are among the most effective tools for second-language acquisition. They allow learners to encounter words in context and develop an intuitive grasp of increasingly complex grammatical forms. Krashen's Input Hypothesis (1985) argues that language acquisition occurs when learners receive comprehensible input slightly above their current level (*i+1*). Graded readers are designed to provide exactly this.

Most quality graded readers exist in paperback or ebook form, accompanied by comprehension questions and lists of target vocabulary. They work. But they present friction for both the writer and the learner.

For the **writer**, creating quality material at a specific CEFR level is slow and expensive. The author must craft an engaging story while operating within strict thematic, lexical and syntactic constraints. A text just one level above a student's ability --- with unfamiliar grammar and vocabulary --- can derail comprehension entirely. Nation (2006) established that learners need 98% known-word coverage for unassisted reading comprehension, a threshold confirmed experimentally by Hu & Nation (2000). Waring's graded reader design principles operationalise this as a maximum of 2 unknown words per 100 running words. Meeting these constraints while maintaining narrative quality requires significant expertise and effort.

For the **learner**, the process of extracting full value from a graded reader has historically been fragmented. Read the text, answer comprehension questions, write down new vocabulary, review it later with a separate tool --- a notebook, physical flashcards, or a spaced-repetition app like Anki. The reading experience and the learning loop are disconnected.

Some apps have begun to close this gap, presenting stories at specific levels with tap-to-translate, built-in vocabulary saving, and audio narration. But these apps tend to be linguistically unreliable in their difficulty targeting, limited or formulaic in their content, and expensive.

This project started by asking whether a multi-agent LLM pipeline can generate stories that are both **engaging and reliably difficulty-controlled** --- producing graded fiction at specific CEFR levels, with each story verified by an ML classifier. The longer-term vision is a constantly evolving inventory of level-controlled content that feeds into an interactive reader application with tap-to-translate, vocabulary review, and audio narration.

**This is a work in progress.** The pipeline generates readable, engaging fiction, but I have not yet achieved output that reliably sits at a specific CEFR level. A2-targeted prose often lands in the A2/B1 border zone; B2-targeted prose oscillates between B1 and C1. Investigating why led to one of the project's most interesting findings: CEFR was designed as a communicative proficiency framework, not a text-complexity measurement system. Its level boundaries are inherently imprecise for text classification --- a problem that no classifier architecture can fully resolve. The project has since evolved toward using vocabulary-frequency bands (NGSL headword counts) as the primary control mechanism, with CEFR retained as the learner-facing label. This shift, and the findings that motivated it, are documented in detail in [Why Classification Is Hard](#why-classification-is-hard----and-how-much-is-fixable) and [Limitations](#limitations).

Malik et al. (2024) showed that while LLMs can be prompted to produce text at specific CEFR levels, reliability varies significantly and tends to degrade at lower levels where constraints are tightest. My findings confirm this: prompt calibration --- how you frame the creative task to the model --- is the primary lever for difficulty control, and it matters at least as much as the linguistic rules you impose.

---

## How It Works

The pipeline uses three LLM-powered agents and one ML classifier, each with a distinct role:

```
                         ┌─────────────────┐
                         │   Story Seed     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    DIRECTOR      │  Plans series arc, episode structure,
                         │                  │  vocabulary targets, continuity rules
                         └────────┬────────┘
                                  │
                          episode_plan.json
                          + story_bible.json
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     WRITER       │  Produces CEFR-constrained prose
                         │                  │  from the episode plan
                         └────────┬────────┘
                                  │
                               prose.txt
                                  │
                    ┌─────────────┼─────────────┐
                    ▼                             ▼
           ┌───────────────┐             ┌───────────────┐
           │  CLASSIFIER   │             │ STATE MANAGER  │
           │ (DistilBERT or│             │                │
           │  SVM + TF-IDF)│             │ Updates story  │
           │               │             │ bible: chars,  │
           │ Predicts CEFR │             │ items, threads,│
           │ level + diag- │             │ locations,     │
           │ nostic feed-  │             │ vocabulary     │
           │ back          │             │                │
           └───────┬───────┘             └───────┬───────┘
                   │                             │
                   │    If actionable            │
                   │    violations:         story_bible_after.json
                   │    retry with               │
                   │    feedback                  │
                   └──────────┐                   │
                              ▼                   ▼
                    ┌─────────────────────────────────┐
                    │  Next episode (loop until done)  │
                    └─────────────────────────────────┘
```

### Director

The Director takes a story seed (or the current story bible for episodes 2+) and produces a structured JSON output containing: a series-level arc plan, an episode plan with key events and vocabulary targets, and continuity metadata (character locations, item states, unresolved threads). The Director is the most prompt-heavy agent --- it encodes reveal budgets, object continuity rules, character introduction constraints, and thread management limits. It requires a model with strong reasoning capabilities; cheaper models often fail to produce valid JSON or exhaust their token budget on thinking without outputting the structured plan.

### Writer

The Writer receives the episode plan, a continuity packet (character states, recent episode history, unresolved threads, last scene position), and level-specific linguistic constraints. It produces prose. The Writer prompt went through 7 versions at A2 and 8 at B2 during development. See [Development Process](#development-process) and [Key Findings](#key-findings) for what changed and why.

### Classifier

After the Writer produces prose, a classifier predicts the CEFR level. Two classifiers are available: a fine-tuned DistilBERT model (6-class, 70% macro-F1) and a Calibrated Linear SVC with dual TF-IDF features (4-class, 69% macro-F1). A separate diagnostic layer extracts 14 hand-crafted linguistic features and compares them against reference bands for the target level. If the diagnostic layer identifies specific feature violations or above-band vocabulary, it provides feedback and the Writer can retry. If the classifier predicts the wrong level but the diagnostic layer finds no specific feature-level issues, the prose is accepted --- retrying without specific guidance produced worse prose in testing.

A key limitation of the current feedback mechanism is granularity: diagnostics operate at the document level ("your average sentence length is too high") rather than identifying *which* sentences or *which* words are problematic. The lexical diagnostic does flag individual above-band words, but feature-level feedback lacks sentence-level targeting. An open question is whether moving to sentence- or paragraph-level diagnostics (e.g., via sliding-window feature extraction) would enable more effective targeted revision --- either by the Writer itself or by a dedicated Editor agent that rewrites only the problematic spans.

### State Manager

The State Manager reads the accepted prose and updates the story bible: character locations, emotional states, key items, episode history, unresolved threads, vocabulary taught, and last scene position. It also performs basic continuity checks --- flagging repeated conversations, characters knowing facts they didn't learn on-page, and thread budget violations. The updated story bible becomes the input for the next episode's Director call.

---

## Difficulty Control

Difficulty control operates at three layers: **planning constraints** (Director), **prose constraints** (Writer), and **post-hoc verification** (Classifier). The pipeline currently uses CEFR level labels throughout, but the actual control mechanisms are largely vocabulary-frequency-based --- the NGSL headword ceilings in the Director prompt are doing most of the work.

### Planning constraints (Director)

The Director prompt encodes level-specific rules:

| Dimension | A2 | B2 |
|---|---|---|
| **Vocabulary ceiling** | ~600--700 headwords, NGSL top 1,000 | ~1,800--2,000 headwords, NGSL top 2,500 |
| **New words per episode** | 5--7 | 8--12 |
| **Grammar scope** | Present/past simple, basic modals (can, must, will), simple if-clauses | Full tense range, passive, relative clauses, reported speech, complex modals, contrast linkers |
| **Sentence length** | 8--9 words average | 10--15 words average |
| **Episode length** | 500--800 words | 600--800 words |
| **Narrative complexity** | Physical stakes, immediate problems, max 2 characters per episode | Moral dilemmas, psychological tension, competing motivations, up to 3 characters |
| **Vocabulary register** | Concrete, physical, story-functional | Concrete with moderate abstraction; at most 1--2 abstract items per episode |

### Prose constraints (Writer)

The Writer receives a compact constraint profile. The current A2 Writer prompt frames constraints as positive performance targets:

- **A2**: "Write in clear past simple. Use short, common connecting words. Describe actions and physical details. If you catch yourself writing a complex sentence, split it."
- **B2**: "Past simple is the backbone; use past perfect and past continuous where they arise naturally. Passive, reported speech, and relative clauses are available --- use them when they make the sentence clearer, not to demonstrate range."

Earlier versions used prohibition lists ("no passive voice, no perfect tenses, no relative clauses") which produced grammatically compliant but stilted prose. Whether the positive framing produces better *CEFR adherence* is unproven --- the classifier results across prompt versions show no clear improvement (see [Finding 2](#2-prohibition-lists-vs-positive-targets)). What is clear is that the prose reads more naturally. The question of whether that naturalness comes at a cost to level control, or whether the classifier is simply unable to detect the difference due to domain mismatch, remains open.

Grammar targets at B2 are framed as **allowed structures**, not required checklist items. This shift was motivated by B2 episodes consistently overshooting into C1 when grammar was treated as a performance checklist (see [Finding 3](#3-grammar-targets-as-checklists-cause-overshoot)).

### Post-hoc verification (Classifier)

The classifier provides a diagnostic signal, not a gate. It runs after every Writer attempt and produces:

1. **Level prediction** with class probabilities
2. **Feature-level diagnostics** --- the top 3 features deviating from the target level's reference bands (e.g., "avg_sent_len=16.2, A2 typical: 7.5--9.8, too high --- simplify")
3. **Lexical diagnostics** --- content words above the target level's vocabulary frequency band

A retry mechanism exists in the pipeline: if the classifier identifies specific actionable violations (feature deviations or above-band vocabulary not assigned by the Director), it constructs diagnostic feedback and triggers a Writer retry. If the prediction is wrong but no specific feature violations are found, the prose is accepted. This policy emerged from testing where blind retries (without specific guidance) degraded prose quality.

In practice, the classifier's domain mismatch means it frequently predicts the wrong level without identifying actionable feature violations, so most prose is accepted on the first attempt.

---

## The Classifier

### Architecture

The classifier uses a two-layer design:

**Layer 1 --- Classification:** A Calibrated Linear SVC (support vector classifier with Platt scaling for probability estimates) trained on dual TF-IDF features (word-level and character-level n-grams). This predicts the CEFR level and produces calibrated class probabilities.

**Layer 2 --- Diagnosis:** 14 hand-crafted linguistic features extracted via spaCy, compared against reference bands (IQR per level per feature) computed from the training corpus:

| Feature | What it measures | Direction |
|---|---|---|
| `avg_sent_len` | Average tokens per sentence | Higher = more complex |
| `avg_word_len` | Average character length of content words | Higher = more complex |
| `ngsl_800/1000/2000/3000` | Proportion of lemmas in NGSL frequency bands | Higher = simpler |
| `out_of_ngsl` | Proportion of lemmas outside NGSL top 3,000 | Higher = more complex |
| `subord_ratio` | Subordinate clause dependencies per token | Higher = more complex |
| `relative_clause_rate` | Relative clauses per sentence | Higher = more complex |
| `passive_rate` | Passive constructions per sentence | Higher = more complex |
| `perfect_rate` | Perfect tense constructions per sentence | Higher = more complex |
| `modal_diversity` | Count of distinct modal lemmas | Higher = more complex |
| `conditional_rate` | Conditional clauses per sentence | Higher = more complex |
| `connector_tier` | Average sophistication tier of discourse connectors | Higher = more complex |
| `ttr` | Type-token ratio | Higher = more complex |

Connector tiers are CEFR-aligned: Tier 1 (A2) includes *and, but, or, so, because*; Tier 2 (B1) adds *although, however, while, unless*; Tier 3 (B2+) adds *furthermore, nevertheless, whereas, moreover*.

The diagnostic layer is direction-aware: if the text is predicted as too complex for the target, it only flags features pushing toward greater complexity. Features below the target band (simpler than typical) are not flagged, since for graded readers, simpler-than-expected is acceptable.

### Training data and performance

Both classifiers were trained on the same combined corpus: four document-level subsets from UniversalCEFR (`elg_cefr_en`, `cambridge_exams_en`, `cefr_asag_en`, `icle500_en`) plus a Kaggle CEFR-labeled text dataset --- 3,324 documents total, primarily academic and learner exam text.

| Model | Classes | Macro-F1 | Strengths | Weaknesses |
|---|---|---|---|---|
| **Calibrated Linear SVC** (TF-IDF word + char n-grams) | 4 (A2--C1) | 69% | Fast inference, no GPU needed, interpretable via feature weights | No contextual understanding, bag-of-words only |
| **Fine-tuned DistilBERT** | 6 (A1--C2) | 70% | Contextual embeddings, covers full CEFR range, strong at extremes (A1: 84% F1, C2: 76% F1) | Needs PyTorch, truncates at 512 tokens, struggles in the B1/B2/C1 middle band (54--63% F1) |

The DistilBERT confusion matrix reveals that misclassification is almost always between adjacent levels (B1↔B2: 53 errors, B2↔C1: 61 errors), never between distant ones (zero A1↔C2 confusion). This is consistent with the ordered nature of CEFR and suggests the model has learned a meaningful difficulty gradient, even where it cannot draw sharp boundaries.

### Why classification is hard --- and how much is fixable

The classifier's main limitation is not total inaccuracy but **boundary uncertainty**. It distinguishes simpler text from more complex text, but adjacent levels are difficult to separate reliably, especially B1/B2 and B2/C1. Three factors contribute, and they are partially independent:

**1. CEFR was not designed to measure text difficulty.** CEFR is a communicative proficiency framework built on teacher consensus and "can-do" descriptors, not a text-complexity measurement system --- something even its creators acknowledged. It describes what a learner can *do* at each level (understand main ideas, interact with fluency), not measurable surface features of text. The boundary between B1 and B2 is not a specific sentence length or vocabulary size --- it is a shift in communicative competence. Two human CEFR raters will disagree on adjacent levels routinely, and MetaMetrics research found that graded readers labeled the same CEFR level by different publishers can differ by 250+ Lexile points in actual text complexity. Expecting a classifier to draw sharp boundaries on a scale that is inherently imprecise is a ceiling on performance that no model architecture can overcome.

More empirically grounded alternatives exist for text leveling: vocabulary-frequency-band control (Nation's 1,000/2,000/3,000-word levels, formalised in the ERF Graded Reader Scale), the Lexile Framework (psychometric, reproducible, places reader ability and text complexity on the same scale), and Pearson's Global Scale of English (GSE, more granular than CEFR but proprietary). The most defensible approach --- and the one this project is converging toward --- is to use CEFR as the surface label (because it is what teachers and learners recognise) while grounding the actual writing and classification decisions in headword counts and word-frequency bands. CEFR earns its place as a communication tool linking generated content into the wider ecosystem of language courses and exams, but treating it as a precise specification for text generation would be a mistake.

**2. Domain mismatch amplifies boundary uncertainty.** Both classifiers were trained on academic and exam text but are applied to AI-generated fiction. These domains have substantially different linguistic profiles:

- Fiction uses concrete nouns (*camera, cupboard, floorboard, stairs*) that are low-frequency in academic corpora but standard at A2 in any ESL curriculum. The TF-IDF features respond to this distributional difference.
- Academic text relies on passive voice, nominalisation, and formal cohesion markers as complexity signals. Fiction at the same CEFR level uses active voice, dialogue, and temporal connectives --- a fundamentally different surface profile.
- Average sentence length is misleading in fiction: dialogue introduces short utterances ("Stop!" / "Where?" / "I don't know.") that pull the mean down, masking the complexity of surrounding descriptive prose.

This domain shift effect is well-documented. Vajjala & Lucic (2018) demonstrated that CEFR classification performance degrades when transferring between text types. Adding DistilBERT improved contextual understanding but did not resolve this --- the model still learns from the same mismatched training distribution.

**3. The generator often overshoots its target.** In testing, some A2-targeted outputs read closer to upper A2 or low B1 on my review (longer descriptive passages, occasional vocabulary above the level). Some B2-targeted outputs appeared closer to C1 (participial phrases, atmospheric description, past perfect). The classifier may be catching real level-control issues in the prompts, not hallucinating errors.

**Practical impact:** Across the A2_001 series (6 episodes), the SVM classifier predicted B1 for 4 of 6 episodes. The A2 probability ranged from 19% to 58%. On my review, the prose sits in the A2/B1 border zone --- the classifier's B1 predictions are not obviously wrong, but nor are they clearly right. Across the B2_001 series (6 episodes), the classifier never predicted B2 --- it oscillated between B1 (3 episodes) and C1 (3 episodes), with B2 probability averaging 21%. On review, the B2 prose leans toward C1 in its complexity, suggesting the prompt overshoot is real.

These results led to two decisions: reducing the acceptance margin from 0.10 to 0.05, and moving the classifier from a hard veto role to an informational diagnostic role. Disentangling how much of the mismatch comes from the classifier versus the generator requires either better training data (fiction-specific CEFR corpora), better prompts (that reliably hit the target), or both.

### Future classifier architecture

Several architectural improvements could address the limitations above:

**Fiction-specific training data.** The single highest-impact change. Candidate corpora include the CLEAR Corpus (4,724 literary/informational excerpts with human difficulty judgments; Crossley et al., 2023), OneStopEnglish (189 articles professionally simplified to three reading levels; Vajjala & Lucic, 2018), and Ace-CEFR (890 conversational texts with fine-grained CEFR labels). Even a small amount of in-domain data mixed into training could substantially reduce the domain shift penalty.

**Ordinal regression.** The current flat multiclass setup penalises an A2→C1 error the same as an A2→B1 error. Ordinal approaches like CORN (Conditional Ordinal Regression; Porwal et al., 2025) or QWK loss (Quadratic Weighted Kappa) exploit CEFR's ordered structure and provide better-calibrated uncertainty for adjacent levels --- exactly the discrimination both classifiers struggle with.

**Complexity contours.** Replace global feature averaging with sliding-window extraction (3--5 sentence windows) to map how complexity varies across a text. Fiction naturally fluctuates --- dense descriptive prose followed by simple dialogue --- and averaging destroys this signal. Feeding complexity contours into a sequence model could improve classification of dialogue-heavy fiction.

**Hybrid transformer + hand-crafted features.** Concatenate DistilBERT's contextual embeddings with pruned hand-crafted features (retaining domain-independent features like tree depth and subordination ratio, dropping domain-dependent ones like passive rate) to combine interpretability with contextual understanding.

---

## Development Process

The pipeline was built over 10 working sessions across one week (March 25--April 1, 2026), starting from a first API call and ending with complete story outputs at two CEFR levels. This section documents the evolution, because the development story itself contains the most useful findings.

### Phase 1: Prompt engineering (Sessions 1--3, March 26--28)

The project started with a single-agent Writer and a story seed. The first 22 experiments systematically varied one prompt dimension at a time (model, temperature, constraint phrasing, max_tokens) and logged every result.

Key discoveries during this phase:

- **Thinking block behaviour.** Claude's extended thinking block functions as a pre-output verification pass: the model drafts the full episode internally, catches constraint violations (passive voice, forbidden modals, advanced vocabulary), and produces a clean version. This is not a bug --- it is genuine CEFR-aware reasoning. But it consumes 50%+ of the output budget and can introduce new violations during self-correction (see [Finding 5](#5-self-correction-can-drift-vocabulary-upward)).
- **Numerical constraints trigger counting.** Any number in the generation prompt --- "8--9 words average", "40--50% dialogue", even "roughly half" --- causes the model to count obsessively in the thinking block, consuming thousands of tokens. This was confirmed across three independent tests (sentence length, dialogue percentage, dialogue "roughly half"). Solution: qualitative instructions for the Writer ("keep sentences short"), numerical measurement in the classifier.
- **The Director--Writer boundary.** The Writer repeats Director descriptions verbatim, even when they're unnatural ("quick eyes" instead of "sharp eyes"). Character descriptions must be written at the target level in the Director prompt, or the Writer must be told it can adapt them.

The three-agent architecture (Director, Writer, State Manager) emerged during this phase, motivated by continuity failures: the Writer alone couldn't track characters, items, and plot threads across episodes.

### Phase 2: Classifier (Session 4, March 31)

Built the CEFR classifier in a single session. Started with Naive Bayes and Logistic Regression on hand-crafted features, then moved to TF-IDF representations. The Calibrated Linear SVC with dual TF-IDF (word-level and character-level n-grams) reached 69% macro-F1, the best of all architectures tested. Added 14 hand-crafted linguistic features as a diagnostic layer and computed reference bands (IQR per level per feature) from the training corpus.

The first pipeline integration test revealed a gap between the classifier's predictions and the target level. Initially attributed entirely to domain mismatch, later review suggested the classifier was partially right --- the A2-targeted prose often sat in the A2/B1 border zone.

### Phase 3: Pipeline integration (Sessions 5--8, March 31--April 1)

Wired the three agents and classifier into a single CLI pipeline with `--plan`, `--generate`, and `--all` modes. Added OpenRouter integration for per-agent model routing (allowing different models for Director, Writer, and State Manager). Built the validator loop: classify, diagnose, provide diagnostic feedback, retry if actionable.

Key integration challenges:
- **State Manager receiving Director's speculative output.** The Director pre-writes a predicted episode summary; if passed to the State Manager, it creates duplicate entries. Fix: the `confirmed_story_bible` pattern --- the State Manager always receives the *previous* State Manager's output, never the Director's speculative version.
- **Vocabulary targets in the feedback loop.** The lexical diagnostic flagged Director-assigned vocabulary as "above band", telling the Writer to replace words it was specifically instructed to use. Fix: filter Director-assigned targets from the diagnostic feedback before passing it to the Writer.

### Phase 4: Level-specific prompts and output (Sessions 9--10)

Generated complete 6-episode series at A2 and B2. The A2 output (A2_001) is usable graded reader material, though several episodes sit in the A2/B1 border zone rather than firmly at A2. The B2 output (B2_001) is readable fiction but tends toward C1 complexity (see [Finding 4](#4-b2-is-the-hardest-level-to-hit)).

Prompt changes during this phase:
- **A2 Writer** reframed from "skilled fiction author" to "plain-language renderer" with performance targets. The prose became more natural, though classifier scores did not measurably change.
- **B2 Writer** stripped of literary/psychological depth framing ("award-winning author known for gripping fiction"). Grammar targets changed from required per-episode structures to allowed features. This reduced C1 overshoot.
- **B2 Director** grammar section changed from "Episode 3 must include: passive modals, perfect infinitive, third conditional" to "these structures are available across the series."

### Phase 5: Classifier upgrade and CEFR reappraisal (Session 11, April 6)

Fine-tuned a DistilBERT classifier on the same training data (6-class, A1--C2), achieving 70% macro-F1. The confusion matrix confirmed that misclassification is almost exclusively between adjacent levels --- the model understands the difficulty gradient but cannot draw sharp boundaries. This, combined with research into how CEFR is used (and misused) in graded reader publishing, led to a reappraisal of the project's framing.

CEFR was designed to describe learner communicative competence, not to measure text complexity. Publisher-assigned CEFR labels are inconsistent (MetaMetrics found 250+ Lexile point differences between books labeled the same CEFR level by different publishers), and the framework's "can-do" descriptors do not map cleanly to measurable text features. The classifier's boundary uncertainty is not purely a technical failure --- it reflects genuine imprecision in the target labels themselves.

This motivated a shift in direction: the pipeline already uses vocabulary-frequency bands (NGSL headword ceilings) as the primary control mechanism, and the findings suggest that leaning further into vocabulary-based metrics --- headword coverage, frequency-band ratios, possibly Lexile scores --- would give both the generator and the classifier more precise, empirically grounded targets. CEFR remains valuable as the learner-facing label that connects generated content to the wider ecosystem of language courses and exams, but treating it as the ground truth for text difficulty was holding the project back.

---

## Key Findings

### 1. Prompt calibration matters more than post-hoc classification

The single biggest lever for CEFR control is how you frame the creative task in the prompt, not how you filter the output afterward. Changing the Writer's identity from "award-winning author known for gripping, character-driven fiction" to "experienced author of graded readers for [level] English learners" produced a noticeable shift in output complexity. The model optimises for whatever the prompt rewards --- when rewarded for literary prestige, it produces literary prestige.

The CaLM work (Malik et al., 2024) supports this: their prompt experiments found that richer CEFR guidance (target-level descriptions, all-level descriptions, few-shot examples) improved proficiency control for GPT-4 compared to simple "write at level X" instructions. Their strongest results came from treating CEFR targeting as a **multi-objective problem** --- balancing level control, text quality, and cost --- rather than as a single yes/no classification gate.

A plausible architecture is **detailed Director, compact Writer, strong validator** --- but I have not yet built a feedback mechanism effective enough to test whether a strong validator can compensate for a lightly steered Writer. The current classifier provides document-level diagnostics that rarely trigger actionable retries, so the "strong validator" component remains aspirational. An alternative approach from the CaLM paper --- **candidate reranking** (generate multiple drafts, classify all, pick the one closest to the target level) --- may be more practical than iterative revision, at the cost of additional generation tokens.

### 2. Prohibition lists vs positive targets

Early Writer prompts specified what was forbidden: "no passive voice, no perfect tenses, no reported speech, no relative clauses." This produced grammatically compliant but stilted prose. Later versions replaced these with positive targets ("write in clear past simple, mostly short sentences, common vocabulary").

The positive framing produced prose that reads more naturally. However, **whether it produces better CEFR adherence is unproven.** Comparing classifier results across prompt versions (A2_001 with earlier prompts vs A2_002/003/004 with positive framing), the classifier predictions did not measurably improve --- the A2 probability remained similar, and most episodes were still classified as B1. This could mean positive framing doesn't help, or it could mean the classifier's domain mismatch makes it unable to detect the difference. The question is open.

### 3. Grammar targets as checklists cause overshoot

The B2 Director initially assigned specific grammar structures per episode ("Episode 3 must include: passive modals, perfect infinitive, third conditional"). This caused the Writer to "perform grammar" --- inserting structures to satisfy the checklist rather than telling the story. The result consistently overshot B2 into C1. Changing grammar targets from **required performance goals** to **allowed features** ("these structures are available; use them when natural") reduced the C1 probability in B2 outputs. This was one of the clearest findings from the B2 development cycle.

### 4. B2 is the hardest level to hit

Across the B2_001 series (6 episodes), the classifier never once predicted B2. Results:

| Episode | Predicted | B2 prob | B1 prob | C1 prob |
|---|---|---|---|---|
| 1 | C1 | 24% | 21% | 51% |
| 2 | B1 | 32% | 37% | 30% |
| 3 | C1 | 17% | 38% | 40% |
| 4 | B1 | 17% | 49% | 32% |
| 5 | B1 | 16% | 58% | 23% |
| 6 | C1 | 20% | 33% | 43% |

B2 probability averaged 21% and never exceeded 32%. On expert review, the B2-targeted prose leans toward C1 complexity, suggesting the generator is genuinely overshooting. But the CaLM paper (Malik et al., 2024) explicitly flags adjacent-level ambiguity as inherent to the task: there is very little confusion between distant levels (A1 vs C2) but substantial uncertainty between neighbouring ones. B2 may be especially hard because it is defined more by what a learner *can do* (understand main ideas of complex text, interact with fluency) than by clear surface-level text features that distinguish it from B1 or C1.

### 5. Self-correction can drift vocabulary upward

The thinking block caught a passive voice violation ("a power cable was coiled on the shelf") and self-corrected to "a power cable lay coiled on the shelf." This fixed the grammar constraint but introduced B2/C1 vocabulary ("lay" as intransitive past of "lie" and "coiled" as a participial adjective). The model optimises for the constraint it's actively checking without cross-referencing vocabulary difficulty.

This means the classifier must catch violations introduced *during self-correction*, not just in the initial draft. It also suggests that more thinking is not always better --- a trade-off exists between grammar compliance and vocabulary control.

### 6. NGSL frequency bands are not CEFR levels

The lexical diagnostic initially used NGSL (New General Service List) frequency rank as a proxy for CEFR level, flagging words outside the top 1,000 lemmas as "above A2." But NGSL is a general English frequency list derived from corpus statistics, not a CEFR syllabus. Common concrete nouns like *camera*, *bedroom*, *cupboard*, *metal*, and *stairs* are standard A2 vocabulary in the Cambridge A2 Key wordlist and the English Vocabulary Profile, but they rank outside the top 1,000 in general English corpora because frequency is dominated by abstract, grammatical, and academic vocabulary.

This produced systematic false positives: the diagnostic told the Writer to replace perfectly appropriate A2 words. The short-term fix was filtering Director-assigned vocabulary targets from the feedback.

The longer-term fix is not to discard NGSL but to use it alongside CEFR-aligned wordlists (Oxford 3000, Cambridge A2 Key, CEFR-J). The two serve different purposes: a CEFR whitelist answers "is this word appropriate for the level?" while NGSL frequency answers "how common is this word in general English?" A word that is CEFR A2 but rank 5,000 in general English is still harder for a learner to *encounter* outside the classroom than a rank 500 word. Both signals matter for readability --- CEFR alignment for level correctness, frequency for practical accessibility.

### 7. Continuity is a planning problem, not a prose problem

Early versions suffered from plot holes, repeated revelations, and characters knowing information they hadn't learned on-page. These were Director failures, not Writer failures. The fix was encoding continuity rules into the Director prompt (object tracking, character foreshadowing, thread budgets, reveal limits) and passing a structured continuity packet to the Writer (last scene position, recent episode history, character states, unresolved threads).

### 8. Any number in a generation prompt triggers counting

This was confirmed across three independent tests. "Average sentence length: 8--9 words" caused the model to count every word in every sentence. "Dialogue: 40--50%" caused 3--5 complete internal drafts. "Roughly half" was treated identically to "50%." The word "roughly" does not prevent counting behaviour.

The rule: **measurement belongs in the classifier (post-generation), not in the generation prompt.** The prompt should describe the *feel* of the target level, not the metrics.

---

## Example Output

Both stories use the same seed: *"A young man gets a weekend job helping to clean out an old, abandoned house. In a dusty cupboard, he finds an old video camera. The tape inside shows the exact room he is standing in, fifty years ago --- and the family is hiding a small metal box under the floorboards."*

### A2 --- Episode 1 (excerpt)

> The house sat at the end of a narrow road. The paint was grey and old. The garden was tall with weeds. Luca put his work bag down and looked at the front door. It was just past eight on Saturday morning. The street was quiet.
>
> Ray Brennan came along the path. He was a big man in a dark coat. He did not smile.
>
> "Here." Ray held out a key. "Everything in this house goes in bags. Throw it all away."
>
> Luca took the key. "Who lived here?" he asked.
>
> "An old family." Ray put his hands in his pockets. "Bag everything. Don't keep anything."
>
> Luca opened his mouth. Ray spoke first.
>
> "Don't keep anything," Ray said again. His voice was flat and clear.

*Classifier: A2 at 43% confidence (B1 at 34%). Short declarative sentences, past simple throughout, concrete physical actions, dialogue with micro-conflict. This episode is solidly A2. Other episodes in the series were classified as B1 --- on review, those sit closer to the A2/B1 border, tending toward low B1.*

### B2 --- Episode 1 (excerpt)

> The house on Crane Street smelled of old wood and damp. Jamie pulled on his work gloves and stepped through the front door, dragging a rubbish bag behind him. The hallway was narrow and dark, with wallpaper peeling away from the plaster in long strips. He had been told the house had stood empty for many years, and it felt like it.
>
> He moved into the front room. A bay window faced the street, its glass filmed with dirt. The fireplace grate was rusted, and beside it an old wooden cupboard stood against the wall, its door hanging slightly open. Jamie started with the easier things --- a broken chair, a stack of damp newspapers, a crate full of nothing worth keeping.

*Classifier: C1 at 51% confidence (B2 at 24%). Longer sentences, past perfect, participial phrases, atmospheric description. On review, the prose does lean toward C1 --- the classifier appears to be catching real overshoot from the B2 prompt rather than misclassifying.*

---

## Research Background

This project draws on several areas of applied linguistics and NLP research:

**Vocabulary coverage and comprehension.** Nation (2006) established that learners need 98% known-word coverage for unassisted reading comprehension. Hu & Nation (2000) confirmed this threshold experimentally. Waring's graded reader design principles operationalise this as a maximum of 2 unknown words per 100 running words. These thresholds directly inform the vocabulary constraints in the Director and Writer prompts.

**The NGSL.** The New General Service List (Browne & Culligan, 2013) provides 2,809 high-frequency word families covering approximately 92% of general English text. I use NGSL frequency bands as one input to the lexical diagnostic, though as noted in [Finding 6](#6-ngsl-frequency-bands-are-not-cefr-levels), frequency rank is not equivalent to CEFR level.

**CEFR-aligned text generation.** Malik et al. (2024), in "From Tarzan to Tolkien" (ACL 2024 Findings), defined the "Proficiency Control Task" with three evaluation dimensions: control error, quality, and cost. Key findings: GPT-4 achieves reasonable CEFR control via prompting alone; richer CEFR descriptions in prompts improve control; adjacent-level ambiguity (A2/B1, B2/C1) is inherent to the task; and top-*k* candidate reranking (generate multiple drafts, select the closest to target) substantially improves control without needing a new generator. Their strongest system, CaLM, uses supervised fine-tuning on synthetic CEFR data with PPO alignment, demonstrating that fine-tuning can outperform prompting for open models. My approach is complementary --- rather than fine-tuning, I structure the generation task through multi-agent orchestration and prompt constraints, with a classifier providing post-hoc diagnostics.

**CEFR text classification.** The classifier draws on established approaches to automatic text difficulty assessment. Vajjala & Lucic (2018) demonstrated that CEFR classification performance degrades significantly under domain shift --- a finding my results confirm directly. The two-layer architecture (TF-IDF classification + hand-crafted feature diagnosis) follows common practice in readability research, where surface features and linguistic features provide complementary signals. Recent work on ordinal approaches to CEFR classification (Porwal et al., 2025; Thuy et al., 2025) suggests that exploiting the ordered structure of CEFR levels with ordinal regression losses (CORN, QWK) can improve both accuracy and calibration, particularly for adjacent-level discrimination.

**Graded reader design.** The project is informed by established principles of graded reader design: controlled vocabulary introduction, grammar staging, narrative engagement at constrained complexity, and systematic recycling of target vocabulary across episodes.

---

## Project Structure

```
cefr-graded-reader/
├── src/
│   ├── pipeline.py                 # Main CLI: --plan, --generate, --all
│   ├── generator/
│   │   ├── prompts.py              # Level-aware prompts (A2, B2) for all 3 agents
│   │   ├── generate.py             # Early experiment runner (22 experiments)
│   │   └── story_bible_schema.json # Story Bible JSON template
│   └── classifier/
│       ├── classify.py             # classify() + diagnose() + lexical_diagnostic()
│       ├── features.py             # 14 hand-crafted linguistic features
│       └── models/                 # Trained model files
│           ├── svc_calibrated.joblib        # SVM baseline
│           └── cefr-distilbert-final/       # Fine-tuned DistilBERT (6-class)
├── data/
│   ├── ngsl/                       # NGSL frequency data
│   └── cefr_dataset/               # Training corpus (UniversalCEFR)
├── outputs/
│   ├── A2_001/                     # Complete 6-episode A2 series
│   │   ├── series_plan.json
│   │   ├── metadata.json
│   │   └── ep_1/ ... ep_6/         # Per-episode: prose, plan, classification, bible
│   ├── B2_001/                     # Complete 6-episode B2 series
│   └── A2_002/, A2_003/, A2_004/   # Additional generation runs (partial)
├── experiments/                    # 22 prompt engineering experiment logs
├── notebooks/
│   ├── 01_data_exploration.ipynb   # SVM classifier training and data exploration
│   └── 02_transformer_classifier.ipynb  # DistilBERT fine-tuning (6-class CEFR)
└── requirements.txt
```

---

## Setup & Usage

### Installation

```bash
git clone https://github.com/billygphillips/cefr-graded-reader.git
cd cefr-graded-reader
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# For the DistilBERT classifier, download model weights
# (config and tokenizer are in the repo; weights must be trained via notebooks/02_transformer_classifier.ipynb)

```

### Configuration

Copy `.env.example` to `.env` and add your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-v1-...
```

### Usage

```bash
# Plan a new series
python src/pipeline.py --plan --level A2 --seed "A young man gets a weekend job..."

# Generate episodes one at a time (auto-detects next episode)
python src/pipeline.py --generate --story outputs/A2_001

# Generate a specific episode (or regenerate it)
python src/pipeline.py --generate --story outputs/A2_001 --episode 3

# Plan and generate all episodes in one run
python src/pipeline.py --all --level A2 --seed "..."

# Use a specific model via OpenRouter
python src/pipeline.py --generate --story outputs/A2_001 --model anthropic/claude-sonnet-4-6

# Use different models per agent
python src/pipeline.py --generate --story outputs/A2_001 \
  --director-model anthropic/claude-sonnet-4-6 \
  --writer-model minimax/minimax-m2.5

# Use the DistilBERT classifier instead of SVM
python src/pipeline.py --generate --story outputs/A2_001 --transformer

# Use Anthropic directly instead of OpenRouter
python src/pipeline.py --generate --story outputs/A2_001 \
  --provider anthropic --model claude-sonnet-4-6
```

Each episode saves to `outputs/{LEVEL}_{NNN}/ep_{N}/` with:
- `episode_plan.json` --- Director's structured plan
- `prose.txt` --- Writer's accepted output
- `classification.json` --- Classifier verdict and probabilities
- `story_bible_after.json` --- Updated story state from State Manager
- `episode_metadata.json` --- Generation metadata (model, tokens, timing)

---

## Limitations

**CEFR as a text-difficulty metric.** CEFR was designed to describe learner communicative competence, not to measure text complexity. Using it as a classification target means training models to predict a label that is inherently imprecise and inconsistent across publishers and raters. The pipeline already uses vocabulary-frequency bands (NGSL) as the primary control mechanism, with CEFR as the surface label --- but the classifier still attempts to predict CEFR directly, which may be the wrong target. A shift toward vocabulary-band coverage, Lexile scores, or ERF headword levels as the primary classification target could improve both accuracy and actionability of the feedback loop.

**Classifier domain mismatch.** Both classifiers (SVM and DistilBERT) were trained on academic/exam text and produce systematic bias on AI-generated fiction. The SVM tends to classify well-written A2 fiction as B1, and never predicts B2 for any B2-targeted episode. The DistilBERT model improves on this slightly (70% macro-F1 across 6 classes vs 69% across 4) but inherits the same domain gap. Both are useful as directional signals but not reliable as authoritative judges of CEFR level.

**NGSL-to-CEFR mapping.** The lexical diagnostic uses NGSL frequency bands as a proxy for CEFR levels. This produces false positives for common concrete nouns that are low-frequency in general corpora but standard in ESL curricula. A proper CEFR-aligned wordlist (Oxford 3000, CEFR-J) would resolve this.

**Single seed tested.** All series were generated from the same story premise. The pipeline has not been tested with diverse genres, settings, or narrative structures.

**No human evaluation.** Output quality has been assessed by me (a qualified ESL teacher with 5+ years of experience) but has not undergone formal human evaluation with learners, other teachers, or independent CEFR assessors.

**Level targeting is unreliable at B2.** The B2 output oscillates between B1 and C1 according to the classifier. This is likely a combination of prompt overshoot, classifier domain mismatch, and the inherent imprecision of CEFR as a text-level boundary.

**Limited prompt variants.** Only A2 and B2 have dedicated prompt profiles. B1 and C1 would need their own prompt variants, constraint tables, and testing.

**No audio generation.** The pipeline produces text only. Audio narration (e.g., via ElevenLabs) is planned for the reader application but not implemented.

**Model dependence.** Prompt quality and constraint adherence varies across models. The Director requires strong reasoning; cheaper models often fail to produce valid JSON. The Writer works best with Claude Sonnet; MiniMax and GLM produced lower-quality or failed outputs in testing.

---

## Future Work

### Classifier improvements

**Better training data.** The most impactful short-term improvement is retraining the classifier on narrative-heavy, CEFR-labeled corpora. Candidate datasets include: the **CLEAR Corpus** (4,724 literary/informational excerpts with human difficulty judgments; Crossley et al., 2023), **OneStopEnglish** (189 articles professionally simplified to three reading levels; Vajjala & Lucic, 2018), the **Cambridge Readability** dataset (331 passages from Cambridge Main Suite exams mapped to CEFR; Xia et al., 2016), and **Ace-CEFR** (890 conversational texts with fine-grained CEFR labels, useful for modelling dialogue; 2025). These could be used to replace or supplement the current UniversalCEFR training set, which is dominated by academic text.

**Ordinal regression.** The current flat multiclass classifier penalises an A2→C1 error the same as an A2→B1 error. Ordinal approaches like CORN (Conditional Ordinal Regression; Porwal et al., 2025) or QWK loss (Quadratic Weighted Kappa) exploit CEFR's ordered structure and provide better-calibrated uncertainty for adjacent levels --- exactly the discrimination this system struggles with.

**Transformer-based classifier (in progress).** A fine-tuned DistilBERT model (6-class, 70% macro-F1) is now integrated as an alternative to the SVM baseline. Next steps: add a CORN ordinal classification head, and experiment with a hybrid approach that concatenates the transformer's contextual embeddings with pruned hand-crafted features (retaining domain-independent features like tree depth and subordination ratio, dropping domain-dependent ones like passive rate) to combine interpretability with contextual understanding.

**Reframing the classification target.** Rather than predicting CEFR levels directly --- which are imprecise and inconsistent as text-difficulty labels --- the classifier could target more empirically grounded metrics: vocabulary-band coverage (percentage of tokens within the top N frequency bands), Lexile scores, or ERF headword levels. CEFR labels could then be derived from these metrics via established cross-walks (e.g., MetaMetrics' Lexile-to-CEFR mapping), giving both a more precise training signal and more actionable diagnostic feedback.

**Feature pruning.** Audit the 14 hand-crafted features for domain robustness. Passive voice rate is highly predictive in academic text but nearly absent in fiction at all levels. Average sentence length is misleading in dialogue-heavy text. These features should be deprioritised or replaced with dialogue-aware variants.

**Synthetic training data.** Generate labeled training data from the pipeline's own outputs, validated by human assessment. This addresses the domain gap directly but requires careful mitigation of circularity --- a classifier trained on its own generator's output may learn to score LLM style rather than CEFR difficulty. Active inheritance (constrained generation with CEFR-aligned vocabulary lists) and MLM-based data augmentation on gold-standard human texts are safer strategies.

**Complexity contours.** Replace global feature averaging with sliding-window extraction (3--5 sentence windows) to map the rhythm of complexity across a text. Fiction naturally fluctuates --- dense descriptive prose followed by simple dialogue --- and averaging these destroys the signal. Feeding complexity contours into a sequence model could improve classification of dialogue-heavy fiction.

### Generation improvements

**Thinking block ablation.** The extended thinking block consumes 50%+ of output tokens and can drift vocabulary upward during self-correction (see [Finding 5](#5-self-correction-can-drift-vocabulary-upward)). The CaLM researchers likely did not use thinking-enabled models. Testing generation without extended thinking may improve level control while substantially reducing cost --- the self-correction that happens in the thinking block may be counterproductive if it introduces vocabulary above the target level.

**Candidate reranking.** The CaLM paper found that generating multiple candidates and selecting the one with the lowest control error substantially improved proficiency targeting. This is simpler than the current retry-with-feedback approach and avoids the problem of degraded prose on revision. Generate 3--5 drafts per episode, classify all, select the best match.

**CEFR-aligned vocabulary system.** Replace NGSL frequency bands with the Oxford 3000 (organised by CEFR level), the Cambridge A2 Key vocabulary list, and CEFR-J vocabulary and grammar profiles. Retain NGSL as a complementary frequency/readability signal.

**Richer CEFR descriptions in Writer prompts.** The CaLM paper found that richer CEFR guidance (target-level descriptions, all-level descriptions, few-shot examples) improved control compared to simple "write at level X" instructions. The current Writer prompt uses constraint lists but does not include a prose description of what the target level *feels like* to read, or examples of correctly-leveled text.

**Vocabulary audit after planning.** Score Director-assigned vocabulary targets on concreteness, register, and CEFR alignment before passing them to the Writer. Flag abstract or literary-register vocabulary that is likely to push prose above the target level.

**CEFR critic and targeted rewriter.** A post-generation agent that identifies specific spans to simplify (abstract nouns, compressed explanations, figurative language, register drift) without rewriting acceptable prose. This moves correction from "regenerate the whole episode" to "fix the specific sentences that are off-level."

**Narrative non-fiction as a fallback.** If creative fiction constraints prove too limiting at certain levels --- if the tension between engaging narrative and strict CEFR compliance cannot be resolved --- narrative non-fiction (explanations of real places, events, processes) may provide a more naturally constrained genre. Non-fiction at A2 has an established tradition in graded reader publishing and may be easier for LLMs to produce within constraints.

**B1 and C1 prompt variants.** The architecture supports arbitrary levels; only the prompt constraints and classifier reference bands need to be defined.

**Multi-model comparison.** Systematic comparison of generation quality across models (Claude Sonnet, DeepSeek R1, Gemini Flash, MiniMax, GLM) measuring CEFR accuracy, prose quality, generation speed, and cost.

### Application

**Interactive reader app.** A web or mobile application that presents generated stories with tap-to-translate, vocabulary saving, spaced-repetition review, and audio narration --- closing the full loop from generation to learner engagement.

---

## Acknowledgements

Built by Billy Phillips as a portfolio project during the NLP course of a Master of Computer Science (AI specialisation) at Monash University. The project was developed with assistance from Claude (Anthropic).

## License

MIT
