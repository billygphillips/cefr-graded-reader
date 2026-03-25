# CEFR Graded Reader — Project TODO

**Last updated:** March 25, 2026 (evening)
**Application deadline:** April 1-2, 2026
**Status:** Planning complete, ready to start building

---

## Session Log

### Session 1 — March 25 (today)
- [x] Research: SLA literature, graded reader design, CEFR features, AI generation
- [x] Research saved to Obsidian vault (Graded-Reader-Design-Research.md)
- [x] Project plan created (Graded-Reader-Project-Plan.md)
- [x] Project folder created (~/Projects/cefr-graded-reader/)
- [x] CLAUDE.md written (AI assistant instructions)
- [x] TODO.md written (this file)
- [ ] Project setup (Python, virtualenv, git init, .gitignore, folder structure)

---

## Phase 1: Story Generation (Thu Mar 26 – Sat Mar 28 morning)

### Setup (do first)
- [ ] Check Python version, install if needed
- [ ] Create virtual environment
- [ ] git init, create .gitignore, first commit
- [ ] Create GitHub repo (billygphillips/cefr-graded-reader), push
- [ ] Create .env.example and .env (API keys)
- [ ] Install initial dependencies (requirements.txt)
- [ ] Create project folder structure (src/, data/, notebooks/, outputs/, tests/)

### API basics
- [ ] Learn: what is an API call, how authentication works, what .env files are for
- [ ] Install anthropic SDK (or openai SDK / openrouter setup)
- [ ] Write first API call — just get a response back from a model
- [ ] Learn: structured JSON output from LLMs

### Prompt engineering
- [ ] Design CEFR level prompt templates (A2, B1, B2)
- [ ] Include CEFR descriptors, vocab constraints, grammar guidelines
- [ ] Separate prompts for fiction vs non-fiction
- [ ] Story architect prompt (outline generation)
- [ ] Creative writer prompt (prose from outline)
- [ ] Test prompts manually, iterate

### Story generation pipeline
- [ ] Model provider abstraction (call different models through one interface)
- [ ] Architect: generate outline (characters, plot, vocab targets)
- [ ] Writer: generate prose from outline at target CEFR level
- [ ] Post-story: comprehension questions + grammar notes + key vocabulary
- [ ] Save output as JSON with full metadata
- [ ] Test: generate a few stories at different levels, inspect quality

### Multi-model comparison
- [ ] Generate same outline with 3-4 different models
- [ ] Compare: narrative quality, CEFR compliance, vocabulary profile, cost
- [ ] Document in notebook (01_model_comparison.ipynb)

---

## Phase 2: CEFR Classifier (Sat Mar 28 – Sun Mar 29)

### Data
- [ ] Download NGSL wordlists (by frequency band)
- [ ] Download Universal CEFR labeled dataset
- [ ] Explore dataset: how many examples per level? What does the data look like?
- [ ] Clean and prepare train/test split

### Feature extraction
- [ ] Install and learn spaCy basics
- [ ] NGSL vocabulary coverage (% words in each frequency band)
- [ ] Average sentence length
- [ ] Type-token ratio
- [ ] Syntactic complexity (parse tree depth, dependency counts)
- [ ] POS distribution
- [ ] Complex connector proportion

### Classifiers
- [ ] Learn scikit-learn basics
- [ ] Naive Bayes classifier (multinomial)
- [ ] Logistic Regression classifier (with regularisation)
- [ ] Evaluate both: confusion matrix, per-class F1, macro-F1
- [ ] **PyTorch NN** — may need to do Module 3 exercise first
  - [ ] Simple 2-3 layer feedforward network
  - [ ] Train and evaluate
- [ ] Compare all three in notebook (02_classifier_training.ipynb)
- [ ] Analyse: which levels get confused? Why?

### Text analysis module
- [ ] Given any text → produce CEFR analysis report
- [ ] Predicted level + confidence
- [ ] Vocabulary coverage breakdown
- [ ] Sentence complexity metrics
- [ ] Flagged out-of-band vocabulary

---

## Phase 3: Integration + Polish (Mon Mar 30 – Tue Mar 31)

### Feedback loop
- [ ] Wire: generate → classify → accept or regenerate
- [ ] If off-target, regenerate with specific feedback to the model
- [ ] Log all attempts (accepted + rejected = training data)
- [ ] Demo: generate stories at each level, show the loop working
- [ ] Pipeline demo notebook (03_pipeline_demo.ipynb)

### README (research report)
- [ ] Project overview and motivation
- [ ] Architecture diagram
- [ ] Research background (CEFR, Nation, NGSL, i+1, graded reader design)
- [ ] Story generation approach (multi-model, prompt engineering)
- [ ] Model comparison results (table + analysis)
- [ ] Classifier comparison results (table + confusion matrices)
- [ ] Example generated stories at each level
- [ ] The feedback loop and why it matters
- [ ] Connection to current research (Tarzan-to-Tolkien, alignment drift)
- [ ] Future work section (see below)
- [ ] Connection to MPI Project 6

### Future work (described in README, not built)
- [ ] RL for CEFR control (CALM-style reward-driven generation)
- [ ] Vector DB story bible for serialised fiction consistency
- [ ] Fine-tuned open model (Llama/Qwen) on generated training data
- [ ] Teacher-learner agent simulation
- [ ] Tap-to-translate, spaced repetition, user progress tracking
- [ ] Audio generation (ElevenLabs)
- [ ] Frontend app

### Stretch goals (only if time)
- [ ] ElevenLabs audio for a few stories
- [ ] Vector DB sketch (Chroma) for story world consistency
- [ ] Teacher-learner simulation design document or early code

---

## Phase 4: Application (Wed Apr 1)

- [ ] Final README polish and push
- [ ] GitHub profile README update with project link
- [ ] CV written and formatted as PDF
- [ ] Form answers drafted (4 text fields)
- [ ] Motivation letter finalised with GitHub link
- [ ] Submit application via portal
- [ ] Email to Levin Brinkmann
- [ ] Follow up on Monash transcript

---

## Blocked / Waiting

- Monash transcript — requested, waiting for delivery
- PyTorch readiness — may need Module 3 exercise before NN classifier phase

---

## Notes for Next Session

_Update this section at the end of each work session so you know where to pick up._

**Next step:** Project setup — check Python, create virtualenv, git init, folder structure.
