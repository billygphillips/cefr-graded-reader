"""
pipeline.py — Two-phase modular CEFR story generation pipeline

Phase 1 (--plan): Director creates a series bible and story plan, saved to outputs/{LEVEL}_{NNN}/
Phase 2 (--generate): Generates one episode at a time with full checkpointing

Per-agent model routing (OpenRouter defaults):
  Director     → z-ai/glm-5          (planning, reasoning, long-horizon arc design)
  Writer       → minimax/minimax-m2.5 (cheap creative writing, storytelling)
  State Manager → qwen/qwen3-14b      (instruction following, JSON extraction)

Override any or all with --director-model / --writer-model / --state-manager-model.
Use --model to set all three to the same model (e.g. for a quick test).

Usage:
  python src/pipeline.py --plan --level A2 --seed "..."
  python src/pipeline.py --generate --story outputs/A2_001
  python src/pipeline.py --generate --story outputs/A2_001 --episode 3
  python src/pipeline.py --generate --story outputs/A2_001 --director-model z-ai/glm-5-turbo
  python src/pipeline.py --generate --story outputs/A2_001 --provider anthropic --model claude-sonnet-4-6
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import requests
import anthropic
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from generator.prompts import (get_director_prompt, get_writer_prompt, get_state_manager_prompt,
                               get_director_version, get_writer_version, get_state_manager_version)
from classifier.classify import load_classifier, classify

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_PROVIDER       = "openrouter"
DEFAULT_DIRECTOR_MODEL = "z-ai/glm-5"          # Strong reasoning for arc planning
DEFAULT_WRITER_MODEL   = "minimax/minimax-m2.5" # Cheap creative writing
DEFAULT_SM_MODEL       = "minimax/minimax-m2.5" # JSON extraction — Qwen3 doesn't reliably output bare JSON
OUTPUTS_DIR            = Path("outputs")

# ── Unified API caller ────────────────────────────────────────────────────────

def call_llm_api(provider, model, max_tokens, system_prompt, user_message):
    """
    Unified LLM API caller. Returns (content_text, stop_reason).

    provider="anthropic": uses the Anthropic SDK directly (claude-sonnet-4-6, etc.)
    provider="openrouter": uses OpenRouter chat completions API (OpenAI-compatible)
    """
    if provider == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return response.content[0].text, response.stop_reason, usage

    elif provider == "openrouter":
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        stop_reason = data["choices"][0]["finish_reason"]
        raw_usage = data.get("usage", {})
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
        }
        return content, stop_reason, usage

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'anthropic' or 'openrouter'.")


def api_call_with_retry(provider, model, max_tokens, system_prompt, user_message,
                        max_retries=3, wait=15):
    """
    Call the LLM API, retrying on transient errors:
    - Anthropic: 529 (Overloaded), 429 (Rate limit)
    - OpenRouter: 429 (Rate limit), 502/503 (Gateway errors)
    """
    retryable = {429, 502, 503, 529}

    for attempt in range(max_retries):
        try:
            return call_llm_api(provider, model, max_tokens, system_prompt, user_message)
            # Returns (content, stop_reason, usage) — callers unpack all three

        except anthropic.APIStatusError as e:
            if e.status_code in retryable and attempt < max_retries - 1:
                print(f"  Anthropic {e.status_code} — waiting {wait}s (retry {attempt + 2}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in retryable and attempt < max_retries - 1:
                print(f"  HTTP {status} — waiting {wait}s (retry {attempt + 2}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  Network error ({e}) — waiting {wait}s (retry {attempt + 2}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


# ── Parsing helpers ───────────────────────────────────────────────────────────

def parse_json_response(raw_output):
    """
    Strip thinking block (Claude: <thinking>, DeepSeek/Qwen: <think>) and code fences,
    then parse and return the JSON dict.

    Tries two passes:
      1. After the closing thinking tag (if present)
      2. The full raw output (catches models that embed JSON inside the thinking block)
    """
    def _extract_json(text):
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            return None

    # Pass 1: text after thinking block
    after_thinking = raw_output
    for close_tag in ["</thinking>", "</think>"]:
        idx = raw_output.find(close_tag)
        if idx != -1:
            after_thinking = raw_output[idx + len(close_tag):]
            break

    result = _extract_json(after_thinking)
    if result is not None:
        return result

    # Pass 2: full raw output (handles models that put JSON inside the thinking block)
    result = _extract_json(raw_output)
    if result is not None:
        return result

    # Both passes failed — print raw output so the user can see what happened
    print("  [ERROR] Could not extract JSON. Raw output (first 3000 chars):")
    print("  " + raw_output[:3000].replace("\n", "\n  "))
    raise ValueError("No JSON object found in model response.")


def extract_prose(raw_output):
    """
    Strip thinking block from Writer output.
    If no thinking block is found, return the raw output as-is.
    """
    for close_tag in ["</thinking>", "</think>"]:
        idx = raw_output.find(close_tag)
        if idx != -1:
            return raw_output[idx + len(close_tag):].strip()
    return raw_output.strip()


# ── Directory helpers ─────────────────────────────────────────────────────────

def next_story_id(level, outputs_dir=OUTPUTS_DIR):
    """Return the next available story ID for a given level (e.g. 'A2_002')."""
    outputs_dir = Path(outputs_dir)
    prefix = f"{level}_"
    existing = [
        d.name for d in outputs_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ] if outputs_dir.exists() else []

    nums = []
    for name in existing:
        suffix = name[len(prefix):]
        if suffix.isdigit():
            nums.append(int(suffix))

    next_num = max(nums) + 1 if nums else 1
    return f"{level}_{next_num:03d}"


def next_episode_number(story_dir):
    """Return the next episode number to generate (1-indexed), based on existing ep_N/ dirs."""
    story_dir = Path(story_dir)
    ep = 1
    while (story_dir / f"ep_{ep}").exists():
        ep += 1
    return ep


# ── Agent functions ───────────────────────────────────────────────────────────

def run_director(story_bible=None, seed=None, level="A2",
                 provider=DEFAULT_PROVIDER, model=DEFAULT_DIRECTOR_MODEL):
    """
    Run the Director agent.

    Episode 1 (--plan mode): pass seed=..., story_bible=None
    Episode N (--generate mode): pass story_bible=<previous bible>, seed=None
    Returns: (episode_plan dict, story_bible dict, call_meta dict)
    """
    system_prompt = get_director_prompt(level)

    if story_bible is not None:
        # Episode 2+: strip vocabulary_introduced (per-episode field, not needed by Director)
        bible_for_director = {
            k: v for k, v in story_bible.items() if k != "vocabulary_introduced"
        }
        user_message = f"<story_bible>{json.dumps(bible_for_director, indent=2)}</story_bible>"
    else:
        user_message = f"<seed>{seed}</seed>"

    t0 = time.time()
    raw_output, stop_reason, usage = api_call_with_retry(
        provider=provider, model=model, max_tokens=20000,
        system_prompt=system_prompt, user_message=user_message,
    )
    duration = round(time.time() - t0, 1)
    print(f"  Director took {duration}s  ({usage['input_tokens']} in / {usage['output_tokens']} out)")

    if stop_reason == "max_tokens":
        print("  WARNING: Director output truncated (max_tokens).")

    director_json = parse_json_response(raw_output)

    for key in ["story_bible", "episode_plan"]:
        if key not in director_json:
            raise KeyError(f"Director JSON missing required key: '{key}'")

    meta = {
        "model": model,
        "provider": provider,
        "prompt_version": get_director_version(level),
        "max_tokens": 12000,
        "stop_reason": stop_reason,
        "token_usage": usage,
        "duration_s": duration,
        "timestamp": datetime.now().isoformat(),
    }
    return director_json["episode_plan"], director_json["story_bible"], meta


def run_writer(episode_plan, confirmed_story_bible,
               provider=DEFAULT_PROVIDER, model=DEFAULT_WRITER_MODEL, level="A2"):
    """
    Run the Writer agent once. Returns (prose string, call_meta dict).
    """
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
                "key_items": c.get("key_items", []),
            }
            for c in confirmed_story_bible["characters"]
        ],
    }
    user_message = (
        f"<continuity_packet>{json.dumps(continuity_packet, indent=2)}</continuity_packet>\n"
        f"<episode_plan>{json.dumps(episode_plan, indent=2)}</episode_plan>"
    )

    t0 = time.time()
    raw_output, stop_reason, usage = api_call_with_retry(
        provider=provider, model=model, max_tokens=16000,
        system_prompt=system_prompt, user_message=user_message,
    )
    duration = round(time.time() - t0, 1)
    print(f"  Writer took {duration}s  ({usage['input_tokens']} in / {usage['output_tokens']} out)")

    if stop_reason == "max_tokens":
        print("  WARNING: Writer output truncated (max_tokens).")

    meta = {
        "model": model,
        "provider": provider,
        "prompt_version": get_writer_version(level),
        "max_tokens": 16000,
        "stop_reason": stop_reason,
        "token_usage": usage,
        "duration_s": duration,
        "timestamp": datetime.now().isoformat(),
    }
    return extract_prose(raw_output), meta


def run_state_manager(prose, episode_plan, confirmed_story_bible, vocabulary_targets,
                      provider=DEFAULT_PROVIDER, model=DEFAULT_SM_MODEL):
    """
    Run the State Manager agent to update the Story Bible from the new episode prose.
    Returns: (updated story_bible dict, call_meta dict)
    """
    system_prompt = get_state_manager_prompt()

    user_message = (
        f"<episode_plan>{json.dumps(episode_plan, indent=2)}</episode_plan>\n"
        f"<current_story_bible>{json.dumps(confirmed_story_bible, indent=2)}</current_story_bible>\n"
        f"<vocabulary_targets>{json.dumps(vocabulary_targets)}</vocabulary_targets>\n"
        f"<new_episode_prose>{prose}</new_episode_prose>"
    )

    t0 = time.time()
    raw_output, stop_reason, usage = api_call_with_retry(
        provider=provider, model=model, max_tokens=16000,
        system_prompt=system_prompt, user_message=user_message,
    )
    duration = round(time.time() - t0, 1)
    print(f"  State Manager took {duration}s  ({usage['input_tokens']} in / {usage['output_tokens']} out)")

    if stop_reason == "max_tokens":
        print("  WARNING: State Manager output truncated (max_tokens).")

    meta = {
        "model": model,
        "provider": provider,
        "prompt_version": get_state_manager_version(),
        "max_tokens": 16000,
        "stop_reason": stop_reason,
        "token_usage": usage,
        "duration_s": duration,
        "timestamp": datetime.now().isoformat(),
    }
    return parse_json_response(raw_output), meta


# ── Continuity validator ──────────────────────────────────────────────────────

def validate_episode_continuity(prose, episode_plan, confirmed_story_bible):
    """
    Rule-based continuity check between Writer output and the episode plan / story bible.
    Returns a list of warning strings (empty if no issues found).
    """
    warnings = []
    prose_lower = prose.lower()

    # Check 1: thread count already exceeds budget
    unresolved = confirmed_story_bible.get("unresolved_threads", [])
    max_threads = (
        confirmed_story_bible
        .get("series_plan", {})
        .get("thread_management", {})
        .get("max_open_threads")
    )
    if max_threads is not None and len(unresolved) > max_threads:
        warnings.append(
            f"Thread count ({len(unresolved)}) already exceeds max_open_threads ({max_threads}) "
            f"before this episode."
        )

    # Check 2: forbidden_moves triggered
    for item in episode_plan.get("forbidden_moves", []):
        # Strip instruction prefix ("Do not X" → check for X)
        check_phrase = item.lower()
        for prefix in ("do not ", "do not repeat ", "do not reveal ", "never "):
            if check_phrase.startswith(prefix):
                check_phrase = check_phrase[len(prefix):]
                break
        if len(check_phrase) > 10 and check_phrase in prose_lower:
            warnings.append(f"Possible forbidden move triggered: '{item}'")

    # Check 3: prose shorter than target (rough check — under 300 words)
    word_count = len(prose.split())
    if word_count < 300:
        warnings.append(
            f"Prose is very short ({word_count} words) — may be incomplete."
        )

    return warnings


# ── Phase 1: Plan a series ────────────────────────────────────────────────────

def cmd_plan(args):
    """
    Create a new series plan.
    Runs the Director once with the seed to build the full Story Bible,
    then saves series_plan.json + metadata.json to outputs/{LEVEL}_{NNN}/.

    Naming: auto-increments to the next available ID (A2_001 → A2_002 etc.).
    If the target directory already exists, skip and print a message (use --force to overwrite).
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    story_id = next_story_id(args.level)
    story_dir = OUTPUTS_DIR / story_id

    # Check for existing series at this level
    existing = sorted([
        d.name for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith(f"{args.level}_")
    ]) if OUTPUTS_DIR.exists() else []

    if existing and not args.force:
        print(f"Existing {args.level} series: {', '.join(existing)}")
        print(f"Use --generate --story outputs/{existing[-1]} to continue the latest series.")
        print(f"Use --force to create a new series ({story_id}) alongside the existing one(s).")
        return

    story_dir.mkdir(parents=True, exist_ok=True)

    director_model = args.director_model or args.model or DEFAULT_DIRECTOR_MODEL

    print(f"\n{'='*60}")
    print(f"PLANNING: {story_id}  ({args.level})")
    print(f"  Director: {args.provider}/{director_model}")
    print(f"{'='*60}")
    print(f"\n--- DIRECTOR (series plan from seed) ---")

    _, story_bible, director_meta = run_director(
        seed=args.seed, level=args.level,
        provider=args.provider, model=director_model,
    )

    # Save Story Bible as series_plan.json.
    # The Director during --plan imagines the world state AFTER Episode 1 (characters have items,
    # are at end-of-episode locations, etc.). We must reset these to the initial pre-Episode-1 state
    # so that --generate for Episode 1 doesn't think Episode 1 already happened.
    story_bible["episode_history"] = []
    story_bible["unresolved_threads"] = []
    story_bible.pop("last_scene_position", None)
    for char in story_bible.get("characters", []):
        char["key_items"] = []
        char.pop("current_state", None)
    series_plan_path = story_dir / "series_plan.json"
    with open(series_plan_path, "w") as f:
        json.dump(story_bible, f, indent=2)

    total_episodes = story_bible.get("series_plan", {}).get("total_episodes", 6)
    title = story_bible.get("metadata", {}).get("title", story_id)

    # Save metadata.json
    metadata = {
        "story_id": story_id,
        "created": director_meta["timestamp"],
        "level": args.level,
        "seed": args.seed,
        "total_episodes": total_episodes,
        "provider": args.provider,
        "director_model": director_model,
        "director_token_usage": director_meta["token_usage"],
    }
    metadata_path = story_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Series '{title}' planned — {total_episodes} episodes.")
    print(f"  series_plan.json → {series_plan_path}")
    print(f"  metadata.json    → {metadata_path}")
    print(f"\nNext: python src/pipeline.py --generate --story {story_dir}")
    return story_dir


# ── Phase 2: Generate an episode ──────────────────────────────────────────────

def cmd_generate(args):
    """
    Generate the next episode (or a specific episode) for an existing series.
    Saves episode_plan.json, prose.txt, classification.json, story_bible_after.json
    to outputs/{story_id}/ep_{N}/.
    """
    story_dir = Path(args.story)

    if not story_dir.exists():
        print(f"Error: story directory not found: {story_dir}")
        sys.exit(1)

    # Load metadata
    with open(story_dir / "metadata.json") as f:
        metadata = json.load(f)

    level          = metadata["level"]
    total_episodes = metadata["total_episodes"]

    # Determine which episode to generate
    if args.episode:
        episode_num = args.episode
        ep_dir = story_dir / f"ep_{episode_num}"
        if ep_dir.exists():
            print(f"Overwriting episode {episode_num} (ep_{episode_num}/ already exists)")
    else:
        episode_num = next_episode_number(story_dir)
        if episode_num > total_episodes:
            print(f"All {total_episodes} episodes already generated for {metadata['story_id']}.")
            return

    ep_dir = story_dir / f"ep_{episode_num}"
    ep_dir.mkdir(exist_ok=True)

    # Resolve per-agent models:
    # Priority: per-agent flag > --model (global override) > per-agent default
    provider        = args.provider
    director_model  = args.director_model  or args.model or DEFAULT_DIRECTOR_MODEL
    writer_model    = args.writer_model    or args.model or DEFAULT_WRITER_MODEL
    sm_model        = args.sm_model        or args.model or DEFAULT_SM_MODEL

    print(f"\n{'='*60}")
    print(f"EPISODE {episode_num}/{total_episodes} — {metadata['story_id']} ({level})")
    print(f"  Director:      {provider}/{director_model}")
    print(f"  Writer:        {provider}/{writer_model}")
    print(f"  State Manager: {provider}/{sm_model}")
    print(f"{'='*60}")

    # Load story bible
    # Episode 1: use series_plan.json (Director output from --plan phase)
    # Episode N: use previous episode's story_bible_after.json
    if episode_num == 1:
        with open(story_dir / "series_plan.json") as f:
            confirmed_story_bible = json.load(f)
    else:
        prev_bible_path = story_dir / f"ep_{episode_num - 1}" / "story_bible_after.json"
        if not prev_bible_path.exists():
            print(f"Error: previous episode bible not found at {prev_bible_path}")
            print(f"Generate episode {episode_num - 1} first.")
            sys.exit(1)
        with open(prev_bible_path) as f:
            confirmed_story_bible = json.load(f)

    # Run Director to get episode plan.
    # Pass the story_bible so Director knows the current series state.
    # For episode 1, episode_history is [] so the Director plans episode 1.
    print(f"\n--- DIRECTOR ---")
    episode_plan, _, director_meta = run_director(
        story_bible=confirmed_story_bible,
        level=level,
        provider=provider,
        model=director_model,
    )

    # Save episode plan
    with open(ep_dir / "episode_plan.json", "w") as f:
        json.dump(episode_plan, f, indent=2)
    print(f"  Episode plan saved.")

    # Run Writer (single call — no retry loop)
    print(f"\n--- WRITER ---")
    prose, writer_meta = run_writer(
        episode_plan, confirmed_story_bible,
        provider=provider, model=writer_model, level=level,
    )

    # Save prose
    with open(ep_dir / "prose.txt", "w") as f:
        f.write(prose)

    # Run continuity validator (rule-based, between Writer and State Manager)
    print(f"\n--- CONTINUITY VALIDATOR ---")
    continuity_warnings = validate_episode_continuity(prose, episode_plan, confirmed_story_bible)
    if continuity_warnings:
        print(f"  {len(continuity_warnings)} warning(s):")
        for w in continuity_warnings:
            print(f"    - {w}")
        print(f"  [NOTE] Review prose manually before using as canon.")
    else:
        print(f"  No issues found.")

    # Classify post-hoc — informational only, never gates or retries
    print(f"\n--- CLASSIFIER (informational) ---")
    print("  Loading classifier...")
    classifier = load_classifier()
    clf_result = classify(prose, classifier)
    print(f"  Predicted: {clf_result['level']}  confidence: {clf_result['confidence']:.2f}  target: {level}")
    print(f"  All probs: {clf_result['probs']}")
    if clf_result["level"] != level:
        print(f"  [NOTE] Predicted level differs from target — review prose manually if needed.")

    with open(ep_dir / "classification.json", "w") as f:
        json.dump(clf_result, f, indent=2)

    # Run State Manager to update story bible
    print(f"\n--- STATE MANAGER ---")
    updated_bible, sm_meta = run_state_manager(
        prose, episode_plan, confirmed_story_bible,
        episode_plan.get("vocabulary_targets", []),
        provider=provider, model=sm_model,
    )

    # Extract and print any continuity warnings from the State Manager's own checks
    sm_warnings = updated_bible.pop("continuity_warnings", [])
    if sm_warnings:
        print(f"  State Manager continuity warnings ({len(sm_warnings)}):")
        for w in sm_warnings:
            print(f"    - {w}")

    # Save updated story bible
    with open(ep_dir / "story_bible_after.json", "w") as f:
        json.dump(updated_bible, f, indent=2)

    # Merge all warnings
    all_warnings = continuity_warnings + sm_warnings

    # Save episode metadata (model, tokens, timing per agent)
    episode_metadata = {
        "story_id": metadata["story_id"],
        "episode": episode_num,
        "level": level,
        "director": director_meta,
        "writer": writer_meta,
        "state_manager": sm_meta,
        "classifier": {
            "predicted": clf_result["level"],
            "target": level,
            "confidence": clf_result["confidence"],
            "probs": clf_result["probs"],
        },
        "continuity_warnings": all_warnings,
        "review_needed": len(all_warnings) > 0,
    }
    with open(ep_dir / "episode_metadata.json", "w") as f:
        json.dump(episode_metadata, f, indent=2)

    # Print a short prose preview
    preview = prose[:300].rstrip() + "..." if len(prose) > 300 else prose
    print(f"\n--- PROSE PREVIEW ---")
    print(preview)

    print(f"\n✓ Episode {episode_num} complete")
    print(f"  Saved to {ep_dir}/")
    print(f"    episode_plan.json      Director's episode plan")
    print(f"    prose.txt              episode prose")
    print(f"    classification.json    classifier verdict (informational)")
    print(f"    story_bible_after.json updated series state")
    print(f"    episode_metadata.json  model/token/timing info per agent")

    if episode_num < total_episodes:
        print(f"\nNext: python src/pipeline.py --generate --story {story_dir}")
    else:
        print(f"\nSeries complete! All {total_episodes} episodes generated.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_all(args):
    """
    Plan a new series and generate all episodes in one run.
    Equivalent to --plan followed by --generate for each episode.
    """
    args.force = True   # allow creation alongside any existing same-level series
    story_dir = cmd_plan(args)
    if story_dir is None:
        return

    with open(story_dir / "metadata.json") as f:
        total_episodes = json.load(f)["total_episodes"]

    args.story = str(story_dir)
    args.episode = None
    args.force = False

    for _ in range(total_episodes):
        cmd_generate(args)


def main():
    parser = argparse.ArgumentParser(
        description="CEFR Graded Reader Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Per-agent OpenRouter defaults:
  Director     : {DEFAULT_DIRECTOR_MODEL}
  Writer       : {DEFAULT_WRITER_MODEL}
  State Manager: {DEFAULT_SM_MODEL}

Examples:
  # Plan a new A2 series (uses Director default model)
  python src/pipeline.py --plan --level A2 --seed "A young man finds a video camera..."

  # Generate next episode (uses per-agent defaults)
  python src/pipeline.py --generate --story outputs/A2_001

  # Regenerate episode 3
  python src/pipeline.py --generate --story outputs/A2_001 --episode 3

  # Use one model for all agents (e.g. for a quick test)
  python src/pipeline.py --generate --story outputs/A2_001 --model minimax/minimax-m2.5

  # Use Anthropic directly (your Claude credits)
  python src/pipeline.py --generate --story outputs/A2_001 --provider anthropic --model claude-sonnet-4-6

  # Mix: Anthropic for Director, cheap OpenRouter for Writer
  python src/pipeline.py --generate --story outputs/A2_001 \\
    --director-model z-ai/glm-5 --writer-model minimax/minimax-m2.5
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan",     action="store_true", help="Plan a new series (Director creates Story Bible)")
    mode.add_argument("--generate", action="store_true", help="Generate next episode for an existing series")
    mode.add_argument("--all",      action="store_true", help="Plan a new series and generate all episodes in one run")

    # --plan options
    parser.add_argument("--level", type=str, default="A2",
                        help="CEFR level: A2, B1, B2, etc. (default: A2)")
    parser.add_argument("--seed",  type=str,
                        help="Story premise / seed (required for --plan)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing series plan (--plan only)")

    # --generate options
    parser.add_argument("--story",   type=str,
                        help="Path to series directory, e.g. outputs/A2_001 (required for --generate)")
    parser.add_argument("--episode", type=int,
                        help="Episode number to generate or regenerate (default: auto-detect next)")

    # Shared: provider + model routing
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER,
                        choices=["anthropic", "openrouter"],
                        help=f"API provider for all agents (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model for ALL agents (overrides per-agent defaults)")

    # Per-agent model overrides (highest priority)
    parser.add_argument("--director-model",      dest="director_model",  type=str, default=None,
                        help=f"Director model (default: {DEFAULT_DIRECTOR_MODEL})")
    parser.add_argument("--writer-model",         dest="writer_model",    type=str, default=None,
                        help=f"Writer model (default: {DEFAULT_WRITER_MODEL})")
    parser.add_argument("--state-manager-model",  dest="sm_model",        type=str, default=None,
                        help=f"State Manager model (default: {DEFAULT_SM_MODEL})")

    args = parser.parse_args()

    if args.plan:
        if not args.seed:
            parser.error("--plan requires --seed")
        cmd_plan(args)
    elif args.all:
        if not args.seed:
            parser.error("--all requires --seed")
        cmd_all(args)
    else:
        if not args.story:
            parser.error("--generate requires --story")
        cmd_generate(args)


if __name__ == "__main__":
    main()
