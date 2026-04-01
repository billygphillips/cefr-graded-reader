# CEFR Story Generator: Continuity and Tone Implementation Spec

This document consolidates the recommended changes for the graded story generator so another implementation agent can apply them directly. The recommendations are based on the current pipeline structure, the current story-state artifacts, and the attached episode outputs. [file:1][file:2][file:3][file:8][file:14][file:25]

## 1. Current diagnosis

The current architecture is sound in broad shape: a Director plans, a Writer drafts prose, and a State Manager updates canonical story state after each episode. The problem is that continuity control is strongest in planning and weakest at the writing handoff. [file:1][file:2]

The Director already sees a rich story bible with `episode_history`, `unresolved_threads`, `last_scene_position`, and a series arc. The Writer, by contrast, currently receives only character summaries plus the current `episode_plan`, which leaves too much room for drift, repetition, and tonal improvisation. [file:1][file:2][file:8][file:14]

The opening episode is strong and grounded: Tom gets a cleaning job, finds a camera, watches a tape, and discovers the hidden box is already gone. Later episodes become less grounded because the story starts stacking large twists and revelations too quickly. [file:5][file:9][file:20][file:25]

The planning layer also contains an internal control issue: the series plan says `max_open_threads` is 5, but the story bible after Episode 2 already lists 6 unresolved threads. That means the system is currently exceeding its own suspense budget early in the series. [file:3][file:14]

## 2. High-priority goals

Implementation should optimize for these goals, in this order: continuity, causal clarity, A2 readability, tonal restraint, then suspense. This priority order fits both the pipeline structure and the strengths of the strongest attached output. [file:1][file:9]

The story should feel like a **small-town mystery**, not a soap-opera thriller. The system should prefer one clear development per episode rather than several dramatic revelations arriving together. [file:9][file:20][file:25]

## 3. Required architecture changes

### 3.1 Strengthen the Writer handoff

The single most important change is to pass the Writer a compact continuity packet derived from the current story bible. At the moment, `run_writer()` does not pass `last_scene_position`, recent episode history, unresolved threads, or current character states into the prompt input. [file:1][file:2]

The Writer should always receive:

- `last_scene_position`
- The last 1 to 2 episode summaries from `episode_history`
- `unresolved_threads`
- Character `current_location`
- Character `current_state`
- Character `key_items`
- A short list of known facts versus suspicions
- Explicit forbidden repeats
- The planned `major_reveal`
- The protagonist's concrete goal for the episode [file:8][file:14]

### 3.2 Make episode plans more explicit

The current `episode_plan.json` format is workable, but it is still too close to a simple event list. The Director should output a tighter control object with start state, end-state intention, reveal budget, and thread accounting. [file:5][file:11][file:20]

Add these fields to every `episode_plan`:

- `goal`
- `start_state`
- `continuity_rules`
- `major_reveal`
- `threads_advanced`
- `threads_resolved`
- `new_thread_created`
- `tone_guardrails`
- `forbidden_moves` [file:3][file:8][file:14]

### 3.3 Add a post-write continuity validator

A lightweight validation step should run after prose generation and before the State Manager finalizes canonical state. This validator can be rule-based at first; it does not need to be a full additional model. [file:1]

The validator should flag:

- Repeated discoveries already marked complete in prior canon.
- Starts that do not logically follow `last_scene_position`.
- Characters knowing something they did not learn on-page.
- Items appearing or disappearing without explanation.
- More than one major revelation when only one was planned.
- New unresolved threads created after the thread budget is full. [file:3][file:8][file:14]

## 4. Prompt package

## 4.1 Director prompt replacement

Use the following as the new Director system prompt or as the core of `get_director_prompt(level)`. It is designed to preserve continuity, control thread count, and reduce melodramatic escalation. [file:1][file:3][file:8][file:14]

```text
DIRECTOR_SYSTEM_PROMPT_V3

You are the Director for a CEFR graded story series.

Your job:
- Plan ONE episode at a time for a 6-part series.
- Output valid JSON only.
- Keep the story coherent, grounded, and easy to follow.
- Prioritize logical continuity over surprise.

Core rules:
1. Respect the existing story bible as canon.
2. Start exactly after `last_scene_position`.
3. Do not repeat scenes, discoveries, or conversations that already happened.
4. Each episode must contain:
   - 1 clear goal for the protagonist
   - 1 obstacle
   - 1 meaningful clue or revelation
   - 1 ending hook
5. Reveal budget:
   - Maximum 1 major revelation per episode
   - Maximum 1 new unanswered question per episode
   - No late melodramatic twists unless seeded earlier in episode_history
6. Thread control:
   - If unresolved_threads is at or above max_open_threads, do not add a new thread
   - Prefer resolving, narrowing, or combining existing threads
7. Tone:
   - Small-town mystery
   - Tense but plausible
   - Serious, restrained, not campy, not soap-opera-like
8. A2 suitability:
   - Events should be easy to explain in simple language
   - Avoid plot turns that require complex backstory dumps
   - Prefer concrete actions: go, look, ask, find, hide, wait, return

Episode planning rules:
- The next episode must feel like a direct consequence of the previous one.
- The protagonist should act on what he learned last time.
- Do not jump ahead emotionally or logically.
- Do not introduce several shocking facts in one episode.
- If a revelation is important, seed it before fully confirming it.

When planning, decide:
- Where the episode starts
- What Tom wants in this episode
- What stops him
- What new fact he learns
- What changes at the end

Output schema:
{
  "story_bible": { ...updated story bible... },
  "episode_plan": {
    "episode_number": <int>,
    "hook": "<one-sentence opening situation>",
    "goal": "<Tom's concrete goal in this episode>",
    "start_state": {
      "location": "<where Tom starts>",
      "time": "<time of day if known>",
      "with_whom": ["..."],
      "items": ["..."],
      "knowledge": ["facts Tom already knows at episode start"],
      "suspicions": ["things Tom suspects but has not confirmed"]
    },
    "continuity_rules": [
      "Must start after last_scene_position",
      "Must not repeat earlier discoveries",
      "Must not contradict known facts"
    ],
    "key_events": [
      "...",
      "...",
      "...",
      "..."
    ],
    "major_reveal": "<single main new fact or advancement>",
    "threads_advanced": ["..."],
    "threads_resolved": ["..."],
    "new_thread_created": "<string or null>",
    "tone_guardrails": [
      "Keep the mystery grounded",
      "No soap-opera reveal stacks",
      "Only one major revelation"
    ],
    "forbidden_moves": [
      "Do not reveal unseeded family-history bombshells",
      "Do not repeat a previous conversation",
      "Do not add extra villains"
    ],
    "vocabulary_targets": ["...", "...", "...", "...", "..."],
    "ending_hook": "<question or danger that leads into next episode>"
  }
}
```

## 4.2 Writer prompt replacement

Use the following as the new Writer system prompt or as the core of `get_writer_prompt(level)`. Its purpose is to make continuity obedience more important than free-form creativity. [file:1][file:8][file:14]

```text
WRITER_SYSTEM_PROMPT_V2

You are the Writer for a CEFR graded reader.

Write prose for exactly one episode.

Your priorities, in order:
1. Continuity
2. Clarity
3. Natural A2 English
4. Suspense
5. Style

Non-negotiable rules:
- Follow the episode_plan exactly.
- Treat all provided facts as canon.
- Start from the given start_state, not from a generic reset.
- Do not repeat any scene or discovery that the plan says already happened.
- Do not invent new major facts, backstory, or twists unless the episode_plan explicitly includes them.
- Do not resolve more than the planned reveal allows.
- Do not add melodramatic reveals just to make the story exciting.

Tone rules:
- Grounded mystery
- Quiet tension
- Serious and believable
- Not silly, campy, exaggerated, or soap-opera-like

A2 language rules:
- Use short, clear sentences.
- Use common words.
- Keep cause and effect obvious.
- Avoid abstract explanations and long speeches.
- Keep dialogue short and purposeful.
- Prefer concrete physical action over explanation.

Scene rules:
- The first paragraph must clearly show where Tom is, what he is doing, and why.
- Every scene must follow naturally from the previous one.
- If Tom learns something important, show how he learns it.
- Each paragraph should move the story forward.
- End with a clear hook, not a random surprise.

Continuity checklist before writing:
- What does Tom already know?
- What does Tom only suspect?
- What item does he physically carry?
- Where is he at the start?
- Who has already spoken to him?
- What must not be repeated?

Output rules:
- Output prose only.
- No headings.
- No bullet points.
- No notes.
- No JSON.
```

## 4.3 State Manager add-on

The State Manager should stay primarily extractive, but it should also record continuity problems instead of silently rewriting canon to match a flawed episode. This is important because the story bible is currently the source of truth for future episodes. [file:1][file:2][file:8][file:14]

```text
STATE_MANAGER_SYSTEM_PROMPT_V2_ADDON

Before outputting the updated story bible, perform these checks:

- Did the prose start after last_scene_position?
- Did it repeat a conversation or discovery that canon says already happened?
- Did any character gain knowledge without an on-page reason?
- Did any item appear, disappear, or move illogically?
- Did the episode add a new unresolved thread when the thread limit was already reached?
- Did the prose introduce any major revelation not present in episode_plan?

If a problem exists:
- Preserve canon
- Do not rewrite canon to fit a continuity mistake
- Add a `continuity_warnings` array to the output
```

## 5. Data contract changes

## 5.1 New Writer input format

Replace the current Writer user message with a structured continuity packet plus the episode plan. This change addresses the current weak handoff in both `generate.py` and `pipeline.py`. [file:1][file:2]

### Current weakness

The existing Writer input is effectively:

```python
user_message = (
    f"{json.dumps(char_profiles, indent=2)}\n"
    f"{json.dumps(episode_plan, indent=2)}"
)
```

This omits critical continuity state that already exists in the story bible. [file:1][file:2]

### Replacement

```python
continuity_packet = {
    "last_scene_position": confirmed_story_bible.get("last_scene_position"),
    "episode_history": confirmed_story_bible.get("episode_history", [])[-2:],
    "unresolved_threads": confirmed_story_bible.get("unresolved_threads", []),
    "characters": [
        {
            "name": c["name"],
            "description": c["description"],
            "flaw": c.get("flaw"),
            "current_location": c.get("current_location"),
            "current_state": c.get("current_state"),
            "key_items": c.get("key_items", [])
        }
        for c in confirmed_story_bible["characters"]
    ]
}

user_message = (
    f"CONTINUITY_PACKET\n{json.dumps(continuity_packet, indent=2)}\n\n"
    f"EPISODE_PLAN\n{json.dumps(episode_plan, indent=2)}"
)
```

## 5.2 Episode plan schema extension

The Director should emit an episode plan closer to this shape:

```json
{
  "episode_number": 3,
  "hook": "Tom returns to the old house after Mrs. Bell's warning.",
  "goal": "Tom wants to learn who was in the house two nights ago.",
  "start_state": {
    "location": "outside the old house",
    "time": "late afternoon",
    "with_whom": [],
    "items": ["old video camera"],
    "knowledge": [
      "A family hid a metal box fifty years ago.",
      "The box is now gone.",
      "Mrs. Bell saw lights in the house two nights ago."
    ],
    "suspicions": [
      "Mrs. Bell knows more than she said.",
      "Mr. Webb may be hiding something."
    ]
  },
  "continuity_rules": [
    "Do not repeat the earlier conversation with Mrs. Bell.",
    "Do not re-discover the empty floorboard.",
    "Tom still has the camera."
  ],
  "key_events": [
    "Tom returns to the house.",
    "He notices a new clue.",
    "The clue points back to Mrs. Bell.",
    "He decides to confront her again."
  ],
  "major_reveal": "Tom finds evidence that Mrs. Bell is connected to the family in the tape.",
  "threads_advanced": [
    "Who took the box?",
    "What happened to the family?"
  ],
  "threads_resolved": [],
  "new_thread_created": null,
  "tone_guardrails": [
    "Keep the episode quiet and tense.",
    "No dramatic confessions yet."
  ],
  "forbidden_moves": [
    "Do not reveal murder backstory.",
    "Do not reveal Tom's father connection."
  ],
  "vocabulary_targets": ["return", "street", "light", "watch", "question"],
  "ending_hook": "Tom sees something in Mrs. Bell's window that changes what he believes."
}
```

## 6. Pipeline implementation changes

## 6.1 `run_writer()` changes

Modify `run_writer()` so the Writer no longer receives only `char_profiles` plus the episode plan. It should receive the continuity packet defined above. [file:1]

Suggested implementation:

```python
def run_writer(episode_plan, confirmed_story_bible,
               provider=DEFAULT_PROVIDER, model=DEFAULT_WRITER_MODEL, level="A2"):
    system_prompt = get_writer_prompt(level)

    continuity_packet = {
        "last_scene_position": confirmed_story_bible.get("last_scene_position"),
        "episode_history": confirmed_story_bible.get("episode_history", [])[-2:],
        "unresolved_threads": confirmed_story_bible.get("unresolved_threads", []),
        "characters": [
            {
                "name": c["name"],
                "description": c["description"],
                "flaw": c.get("flaw"),
                "current_location": c.get("current_location"),
                "current_state": c.get("current_state"),
                "key_items": c.get("key_items", [])
            }
            for c in confirmed_story_bible["characters"]
        ]
    }

    user_message = (
        f"CONTINUITY_PACKET\n{json.dumps(continuity_packet, indent=2)}\n\n"
        f"EPISODE_PLAN\n{json.dumps(episode_plan, indent=2)}"
    )

    raw_output, stop_reason, usage = api_call_with_retry(
        provider=provider,
        model=model,
        max_tokens=12000,
        system_prompt=system_prompt,
        user_message=user_message,
    )

    return extract_prose(raw_output), meta
```

## 6.2 Add a continuity validator step

Add a small validation function between Writer and State Manager. It can be rule-based at first and expanded later. [file:1][file:8][file:14]

Suggested shape:

```python
def validate_episode_continuity(prose, episode_plan, confirmed_story_bible):
    warnings = []

    last_scene = confirmed_story_bible.get("last_scene_position", "")
    unresolved = confirmed_story_bible.get("unresolved_threads", [])
    max_threads = (
        confirmed_story_bible
        .get("series_plan", {})
        .get("thread_management", {})
        .get("max_open_threads")
    )

    if max_threads is not None and len(unresolved) > max_threads:
        warnings.append("Unresolved thread count already exceeds configured maximum.")

    forbidden = episode_plan.get("forbidden_moves", [])
    for item in forbidden:
        if item.lower() in prose.lower():
            warnings.append(f"Possible forbidden move triggered: {item}")

    return warnings
```

This first version is intentionally simple. A better later version can call a small verifier model with the continuity packet and episode plan. [file:1]

## 6.3 Preserve canon when prose is wrong

If continuity warnings are raised, do not let the State Manager silently convert the bad prose into new canon. Instead, either stop the pipeline or write the warnings into episode metadata and require manual review. [file:1][file:8][file:14]

A practical policy:

- Zero warnings: proceed normally.
- One minor warning: proceed, but record warning in `episode_metadata.json`.
- Any major warning: mark episode as review-needed and do not overwrite canonical state without approval. [file:1]

## 7. Tone and reveal policy

The attached outputs show that the mystery works best when the story stays physically concrete and emotionally restrained. Episode 1 is strong because every beat is visible and easy to follow. [file:5][file:9]

The story becomes weaker when it compresses too many revelations into one episode, such as family secrets, murder backstory, inheritance motive, a missing father, and hidden notes all arriving in a short space. That escalation is the main reason the series starts to feel silly. [file:20][file:25]

Adopt these story rules:

- One major revelation per episode.
- One new question per episode at most.
- No “everything changes” episodes before Episode 5.
- Big late twists must be foreshadowed at least twice.
- Avoid hidden-relative reveals unless seeded early.
- Avoid stacking crime, inheritance, secret family links, and dead-parent backstory together. [file:3][file:20][file:25]

### Recommended six-episode rhythm

- Episode 1: discovery of camera and empty hiding place.
- Episode 2: first outside witness and warning.
- Episode 3: new clue links Mrs. Bell to the tape.
- Episode 4: confirmation that Mrs. Bell was the girl in the video.
- Episode 5: Mrs. Bell gives the full but still focused truth about the box and Mr. Webb.
- Episode 6: Tom finds proof and exposes the truth. [file:3][file:8]

This preserves the original arc while reducing melodramatic pile-up. [file:3][file:25]

## 8. Suggested acceptance tests

The implementation agent should consider the change complete only if the system passes these checks:

### Continuity tests

- Episode N always starts directly after `last_scene_position` from Episode N-1. [file:8][file:14]
- An already-completed conversation is not replayed as if new. [file:8][file:14]
- Tom's held items remain consistent across episodes unless changed on-page. [file:8][file:14]
- Characters do not know facts they did not learn in prose. [file:8][file:14]

### Thread tests

- `unresolved_threads` never exceeds configured `max_open_threads` without a warning. [file:3][file:14]
- Each episode either resolves, narrows, or advances at least one existing thread. [file:3][file:14]
- New threads are only added before the configured cutoff. [file:3]

### Tone tests

- No episode contains more than one major revelation unless explicitly allowed by plan. [file:20][file:25]
- The prose remains grounded and avoids sudden soap-opera escalation. [file:9][file:25]
- Dialogue stays short and functional at A2 level. [file:9][file:25]

### Pipeline tests

- Writer input includes continuity packet plus episode plan. [file:1][file:2]
- State Manager outputs `continuity_warnings` when appropriate. [file:1][file:2]
- Episode metadata records continuity-validator warnings. [file:1]

## 9. Implementation order

Recommended order for the other agent:

1. Update the Director prompt. [file:1]
2. Update the Writer prompt. [file:1]
3. Extend the episode plan schema. [file:3][file:5][file:20]
4. Change `run_writer()` to pass continuity packet. [file:1][file:2]
5. Add continuity validator between Writer and State Manager. [file:1]
6. Extend State Manager output with `continuity_warnings`. [file:1][file:2]
7. Regenerate Episodes 2 to 6 and compare for tone and continuity improvement. [file:9][file:25]

## 10. Minimum viable patch

If implementation time is limited, do only these three things first:

- Pass `last_scene_position`, recent `episode_history`, unresolved threads, and character current states into the Writer input. [file:1][file:2][file:8][file:14]
- Enforce “one major revelation per episode” in the Director prompt. [file:3][file:20][file:25]
- Add a validator warning when unresolved threads exceed the configured maximum. [file:3][file:14]

These three changes should produce the biggest immediate gain in continuity and tone without requiring a full rebuild of the pipeline. [file:1][file:3][file:14]
