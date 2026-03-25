# CEFR Graded Reader — AI Assistant Instructions

## About the Developer

Billy is a Master of Computer Science student (AI specialisation, Monash University) and ESL teacher with 5+ years of teaching experience. He is building this project as a portfolio piece for a research internship application at the Max Planck Institute (Center for Humans and Machines, Berlin).

**Current skill level:**
- Python: competent in basics (data structures, functions, OOP basics, file I/O) but has NOT built a project from scratch before
- NLP: understands concepts (n-grams, Naive Bayes, logistic regression, text classification) from coursework but has NOT implemented them in Python outside of guided notebooks
- Math: probability, linear algebra basics, calculus (derivatives, chain rule, gradient descent)
- scikit-learn: treat as NEW — has not used it independently
- spaCy / NLTK: treat as NEW — has not used them independently
- PyTorch: has NOT done PyTorch yet — may need to do Module 3 PyTorch exercise first before the NN classifier phase
- Git: NEW — learning version control through this project
- API calls: treat as NEW — has used Claude API before but doesn't remember how; explain from scratch
- Virtual environments, pip, project structure: treat as NEW

**What he's learning through this project (treat ALL of these as new):**
- How to structure a Python project from scratch
- Virtual environments, pip, requirements.txt
- Git workflow (init, add, commit, push, .gitignore)
- How to call APIs (requests, SDKs, authentication, .env files)
- Prompt engineering for structured output
- Implementing NLP pipelines in Python (spaCy, scikit-learn)
- Feature engineering for text classification
- Training and evaluating ML models
- PyTorch basics (when we get to the NN classifier)
- Multi-model comparison methodology

## How to Teach

- **Explain every decision.** Don't just write code — explain WHY this structure, WHY this library, WHY this pattern. Billy learns by understanding rationale.
- **No code dumps.** Build files together, step by step. Each file should be understood before moving to the next.
- **Connect to theory.** When implementing something (e.g., Naive Bayes classifier), connect it back to what Billy learned in his NLP course. Use familiar terminology.
- **Suggest commits.** Prompt Billy to commit after meaningful chunks of work. Help him write good commit messages.
- **Be concise.** Billy prefers density over length. Don't over-explain things he already knows.

## What NOT to Do

- Don't generate entire files or modules at once — build incrementally
- Don't use advanced Python patterns without explaining them (decorators, metaclasses, complex comprehensions)
- Don't assume familiarity with frameworks beyond what's listed above
- Don't skip error handling explanations — Billy should understand why we handle errors a certain way
- Don't restructure or refactor without explaining the motivation
- Don't add unnecessary complexity — keep it simple, we can refactor later

## Project Overview

A Python CLI pipeline that:
1. **Generates** CEFR-leveled stories (A2, B1, B2) using multiple LLM APIs
2. **Classifies** text difficulty using trained ML models (NB, LR, NN)
3. **Loops** until generated stories match target CEFR level
4. **Compares** multiple generation models and classification approaches
5. **Saves** all outputs as labeled training data for future model fine-tuning

No frontend. This is a research-oriented prototype with a README that reads like a research report.

## Architecture

```
Story Architect (LLM) → outline
        ↓
Creative Writer (LLM) → prose at target CEFR level
        ↓
Post-Story Generator (LLM) → comprehension questions + grammar notes + key vocab
        ↓
CEFR Classifier (NB/LR/NN) → predicted level + vocabulary analysis
        ↓
On target? → ACCEPT + save | Off target? → REGENERATE with feedback
```

All outputs saved as JSON with full metadata (model used, target level, predicted level, etc.) for future training data.

## CEFR Sub-band Targets

Each story targets a single CEFR level. Sub-bands provide numeric targets for generation and classification:

| Sub-band | NGSL coverage | Avg sentence length | New words/episode |
|----------|--------------|--------------------|--------------------|
| A2-low   | 99%+ top 800  | 7-8 words  | 3-5   |
| A2-mid   | 98%+ top 900  | 8-9 words  | 5-7   |
| A2-high  | 97%+ top 1000 | 9-10 words | 7-10  |
| B1-low   | 98%+ top 1500 | 11-12 words | 8-10  |
| B1-mid   | 97%+ top 1800 | 12-14 words | 10-12 |
| B1-high  | 96%+ top 2000 | 14-16 words | 12-15 |
| B2-low   | 97%+ top 2500 | 15-17 words | 12-15 |
| B2-mid   | 96%+ top 2800 | 17-19 words | 15-20 |
| B2-high  | 95%+ top 3000 | 19-21 words | 18-25 |

Key threshold: Nation (2006) — learners need 98% known-word coverage for unassisted comprehension. Max 2 unknown words per 100 running words.

## Key Research References

- Nation (2006) — "How Large a Vocabulary Is Needed For Reading and Listening?"
- Hu & Nation (2000) — 98% vocabulary coverage threshold
- Rahmani et al. (2024) — "From Tarzan to Tolkien" (CALM, CEFR-aligned LLMs via distillation + RL)
- NGSL (Browne & Culligan) — 2,809 word families → 92% coverage of general English
- "Alignment Drift in CEFR-prompted LLMs" (2025) — LLMs drift upward during extended generation
- Waring — graded reader design: "the story, the story, the story"; max 2 unknown words per 100

## Project Structure

```
cefr-graded-reader/
├── CLAUDE.md                  # This file — AI assistant instructions
├── TODO.md                    # Living checklist — update after each session
├── README.md                  # Research report + project documentation (for GitHub)
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── ngsl/                 # NGSL wordlists by frequency band
│   ├── cefr_dataset/         # Universal CEFR labeled dataset
│   └── generated/            # All generated stories (labeled training data)
├── src/
│   ├── generator/            # Story generation pipeline
│   ├── classifier/           # CEFR classification models
│   ├── analysis/             # Vocabulary and readability analysis
│   └── pipeline.py           # Full generate → classify → loop
├── notebooks/                # Jupyter notebooks for exploration
├── outputs/stories/          # Generated story corpus (JSON)
└── tests/
```

## Detailed research

For deeper context on SLA theory, CEFR level characteristics, graded reader design principles, competitor analysis, and modeling strategy, see Billy's Obsidian vault at:
`/Users/billyphillips/Library/Mobile Documents/iCloud~md~obsidian/Documents/Billy's Brain/`

Key files:
- `04-Career/MPI-Research/Graded-Reader-Design-Research.md`
- `04-Career/MPI-Research/Graded-Reader-Project-Plan.md`

These are not in this repo — they're personal research notes.
