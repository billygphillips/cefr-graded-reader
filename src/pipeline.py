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
from generator.prompts import get_director_prompt, get_writer_prompt, get_state_manager_prompt
from classifier.classify import load_classifier, classify, should_accept, diagnose

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_PROVIDER       = "openrouter"
DEFAULT_DIRECTOR_MODEL = "z-ai/glm-5"          # Strong reasoning for arc planning
DEFAULT_WRITER_MODEL   = "minimax/minimax-m2.5" # Cheap creative writing
DEFAULT_SM_MODEL       = "qwen/qwen3-14b"       # Instruction following for JSON extraction
MAX_RETRIES            = 3
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
        return response.content[0].text, response.stop_reason

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
        return content, stop_reason

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
    Strip thinking block (Claude: <thinking>, DeepSeek: <think>) and code fences,
    then parse and return the JSON dict.
    """
    # Remove thinking block if present — check both tag styles
    for close_tag in ["</thinking>", "</think>"]:
        idx = raw_output.find(close_tag)
        if idx != -1:
            raw_output = raw_output[idx + len(close_tag):]
            break

    cleaned = raw_output.replace("```json", "").replace("```", "").strip()
    json_start = cleaned.find("{")
    json_end   = cleaned.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        raise ValueError("No JSON object found in model response.")
    return json.loads(cleaned[json_start:json_end])


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
    Returns: (episode_plan dict, story_bible dict)
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
    raw_output, stop_reason = api_call_with_retry(
        provider=provider, model=model, max_tokens=12000,
        system_prompt=system_prompt, user_message=user_message,
    )
    print(f"  Director took {time.time() - t0:.1f}s")

    if stop_reason == "max_tokens":
        print("  WARNING: Director output truncated (max_tokens).")

    director_json = parse_json_response(raw_output)

    for key in ["story_bible", "episode_plan"]:
        if key not in director_json:
            raise KeyError(f"Director JSON missing required key: '{key}'")

    return director_json["episode_plan"], director_json["story_bible"]


def run_writer(episode_plan, confirmed_story_bible, feedback=None,
               provider=DEFAULT_PROVIDER, model=DEFAULT_WRITER_MODEL, level="A2"):
    """
    Run the Writer agent. If feedback is provided, append it as a classifier correction hint.
    Returns: prose string (thinking block already stripped).
    """
    system_prompt = get_writer_prompt(level)

    char_profiles = [
        {"name": c["name"], "description": c["description"], "flaw": c["flaw"]}
        for c in confirmed_story_bible["characters"]
    ]
    user_message = (
        f"<character_profiles>{json.dumps(char_profiles, indent=2)}</character_profiles>\n"
        f"<episode_plan>{json.dumps(episode_plan, indent=2)}</episode_plan>"
    )
    if feedback:
        user_message += f"\n\n<classifier_feedback>{feedback}</classifier_feedback>"

    t0 = time.time()
    raw_output, stop_reason = api_call_with_retry(
        provider=provider, model=model, max_tokens=12000,
        system_prompt=system_prompt, user_message=user_message,
    )
    print(f"  Writer took {time.time() - t0:.1f}s")

    if stop_reason == "max_tokens":
        print("  WARNING: Writer output truncated (max_tokens).")

    return extract_prose(raw_output)


def run_state_manager(prose, confirmed_story_bible, vocabulary_targets,
                      provider=DEFAULT_PROVIDER, model=DEFAULT_SM_MODEL):
    """
    Run the State Manager agent to update the Story Bible from the new episode prose.
    Returns: updated story_bible dict.
    """
    system_prompt = get_state_manager_prompt()

    user_message = (
        f"<current_story_bible>{json.dumps(confirmed_story_bible, indent=2)}</current_story_bible>\n"
        f"<vocabulary_targets>{json.dumps(vocabulary_targets)}</vocabulary_targets>\n"
        f"<new_episode_prose>{prose}</new_episode_prose>"
    )

    t0 = time.time()
    raw_output, stop_reason = api_call_with_retry(
        provider=provider, model=model, max_tokens=8000,
        system_prompt=system_prompt, user_message=user_message,
    )
    print(f"  State Manager took {time.time() - t0:.1f}s")

    if stop_reason == "max_tokens":
        print("  WARNING: State Manager output truncated (max_tokens).")

    return parse_json_response(raw_output)


# ── Classify-and-retry loop ───────────────────────────────────────────────────

def generate_and_classify(episode_plan, confirmed_story_bible, classifier,
                          target_level="A2", max_retries=MAX_RETRIES,
                          provider=DEFAULT_PROVIDER, model=DEFAULT_WRITER_MODEL):
    """
    Run the Writer in a classify-and-retry loop until the output matches target_level
    or max_retries is exhausted. Passes classifier diagnostic feedback to Writer on retry.

    Returns: (prose, classification_result, attempts_taken)
    """
    feedback = None

    for attempt in range(1, max_retries + 1):
        print(f"\n--- WRITER (attempt {attempt}/{max_retries}) ---")
        if feedback:
            print(f"  Feedback: {feedback}")

        prose = run_writer(
            episode_plan, confirmed_story_bible, feedback=feedback,
            provider=provider, model=model, level=target_level,
        )
        result = classify(prose, classifier)
        predicted, confidence = result["level"], result["confidence"]

        print(f"  Classifier: predicted={predicted}  confidence={confidence:.2f}  target={target_level}")
        print(f"  All probs: {result['probs']}")

        if should_accept(result, target_level):
            print(f"  ✓ Accepted on attempt {attempt}")
            return prose, result, attempt

        if attempt < max_retries:
            feedback = diagnose(prose, target_level, classifier, predicted_level=predicted)

    print(f"  ✗ Max retries reached — accepting last output (predicted={predicted}).")
    return prose, result, max_retries


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

    _, story_bible = run_director(
        seed=args.seed, level=args.level,
        provider=args.provider, model=director_model,
    )

    # Save Story Bible as series_plan.json
    series_plan_path = story_dir / "series_plan.json"
    with open(series_plan_path, "w") as f:
        json.dump(story_bible, f, indent=2)

    total_episodes = story_bible.get("series_plan", {}).get("total_episodes", 6)
    title = story_bible.get("metadata", {}).get("title", story_id)

    # Save metadata.json
    metadata = {
        "story_id": story_id,
        "created": datetime.now().isoformat(),
        "level": args.level,
        "seed": args.seed,
        "total_episodes": total_episodes,
        "provider": args.provider,
        "director_model": director_model,
    }
    metadata_path = story_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Series '{title}' planned — {total_episodes} episodes.")
    print(f"  series_plan.json → {series_plan_path}")
    print(f"  metadata.json    → {metadata_path}")
    print(f"\nNext: python src/pipeline.py --generate --story {story_dir}")


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
    episode_plan, _ = run_director(
        story_bible=confirmed_story_bible,
        level=level,
        provider=provider,
        model=director_model,
    )

    # Save episode plan
    with open(ep_dir / "episode_plan.json", "w") as f:
        json.dump(episode_plan, f, indent=2)
    print(f"  Episode plan saved.")

    # Generate prose (with or without classifier loop)
    if args.no_classify:
        print(f"\n--- WRITER ---")
        prose = run_writer(
            episode_plan, confirmed_story_bible,
            provider=provider, model=writer_model, level=level,
        )
        classification_result = {"level": "skipped", "confidence": 0.0, "probs": {}}
        attempts = 1
        print(f"  ✓ (classifier skipped)")
    else:
        print("\nLoading classifier...")
        classifier = load_classifier()
        prose, classification_result, attempts = generate_and_classify(
            episode_plan, confirmed_story_bible, classifier,
            target_level=level, max_retries=MAX_RETRIES,
            provider=provider, model=writer_model,
        )

    # Save prose
    with open(ep_dir / "prose.txt", "w") as f:
        f.write(prose)

    # Save classification (include model info for the record)
    classification_result["writer_model"] = writer_model
    with open(ep_dir / "classification.json", "w") as f:
        json.dump(classification_result, f, indent=2)

    # Run State Manager to update story bible
    print(f"\n--- STATE MANAGER ---")
    updated_bible = run_state_manager(
        prose, confirmed_story_bible,
        episode_plan.get("vocabulary_targets", []),
        provider=provider, model=sm_model,
    )

    # Save updated story bible
    with open(ep_dir / "story_bible_after.json", "w") as f:
        json.dump(updated_bible, f, indent=2)

    # Print a short prose preview
    preview = prose[:300].rstrip() + "..." if len(prose) > 300 else prose
    print(f"\n--- PROSE PREVIEW ---")
    print(preview)

    print(f"\n✓ Episode {episode_num} complete  ({attempts} writer attempt(s))")
    print(f"  Saved to {ep_dir}/")
    print(f"    episode_plan.json      Director's episode plan")
    print(f"    prose.txt              accepted episode prose")
    print(f"    classification.json    classifier verdict")
    print(f"    story_bible_after.json updated series state")

    if episode_num < total_episodes:
        print(f"\nNext: python src/pipeline.py --generate --story {story_dir}")
    else:
        print(f"\nSeries complete! All {total_episodes} episodes generated.")


# ── CLI ───────────────────────────────────────────────────────────────────────

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

  # Skip classifier (faster, no retry loop)
  python src/pipeline.py --generate --story outputs/A2_001 --no-classify

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
    parser.add_argument("--no-classify", action="store_true", dest="no_classify",
                        help="Skip the classifier feedback loop")

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
    else:
        if not args.story:
            parser.error("--generate requires --story")
        cmd_generate(args)


if __name__ == "__main__":
    main()
