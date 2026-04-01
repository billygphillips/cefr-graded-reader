"""
pipeline.py — Full generate → classify → feedback loop

Runs the three-agent pipeline (Director → Writer → State Manager) with a
classify-and-retry loop around the Writer. If the Writer's output doesn't match
the target CEFR level, the classifier's verdict is fed back to the Writer as a
correction hint, and it tries again (up to MAX_RETRIES times).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
import json
import time
import anthropic

from generator.prompts import DIRECTOR_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT, STATE_MANAGER_SYSTEM_PROMPT
from classifier.classify import load_classifier, classify, should_accept, diagnose

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

TARGET_LEVEL = "A2"
MAX_RETRIES  = 3

SEED = """
A young man gets a weekend job helping to clean out an old, abandoned house.
In a dusty cupboard, he finds an old video camera. The tape inside shows the
exact room he is standing in, fifty years ago — and the family is hiding a
small metal box under the floorboards.
"""

CEFR_ORDER = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# ── Helpers ──────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def api_call_with_retry(func, max_retries=3, wait=15):
    """Call func(), retrying on 529 Overloaded errors up to max_retries times."""
    for attempt in range(max_retries):
        try:
            return func()
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                print(f"API overloaded — waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                raise


def parse_json_response(raw_output):
    """Strip thinking block + code fences, return parsed JSON dict."""
    thinking_end = raw_output.find("</thinking>")
    after_thinking = raw_output[thinking_end + len("</thinking>"):] if thinking_end != -1 else raw_output
    cleaned = after_thinking.replace("```json", "").replace("```", "").strip()
    json_start = cleaned.find("{")
    json_end   = cleaned.rfind("}") + 1
    return json.loads(cleaned[json_start:json_end])


def build_feedback(result, target_level, text, classifier):
    """
    Build a specific diagnostic feedback string for the Writer using
    feature deviations from the target level's reference bands.
    """
    return diagnose(text, target_level, classifier, predicted_level=result["level"])


# ── Agent calls ──────────────────────────────────────────────────────────────

def run_director(story_bible=None):
    """Run Director. If story_bible is None, starts Episode 1 from SEED."""
    if story_bible:
        bible_for_director = {k: v for k, v in story_bible.items() if k != "vocabulary_introduced"}
        user_message = f"<story_bible>{json.dumps(bible_for_director, indent=2)}</story_bible>"
    else:
        user_message = f"<seed>{SEED}</seed>"

    response = api_call_with_retry(lambda: client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=DIRECTOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ))
    if response.stop_reason == "max_tokens":
        print("WARNING: Director output truncated.")

    director_json = parse_json_response(response.content[0].text)

    for key in ["story_bible", "episode_plan"]:
        if key not in director_json:
            raise KeyError(f"Director JSON missing key: '{key}'")

    return director_json["episode_plan"], director_json["story_bible"]


def run_writer(episode_plan, confirmed_story_bible, feedback=None):
    """Run Writer. If feedback is provided, include it as a correction hint."""
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

    response = api_call_with_retry(lambda: client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=12000,
        system=WRITER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ))
    if response.stop_reason == "max_tokens":
        print("WARNING: Writer output truncated.")

    raw = response.content[0].text
    thinking_end = raw.find("</thinking>")
    prose = raw[thinking_end + len("</thinking>"):].strip() if thinking_end != -1 else raw

    return prose


def run_state_manager(prose, confirmed_story_bible, vocabulary_targets):
    """Run State Manager to update the Story Bible."""
    user_message = (
        f"<current_story_bible>{json.dumps(confirmed_story_bible, indent=2)}</current_story_bible>\n"
        f"<vocabulary_targets>{json.dumps(vocabulary_targets)}</vocabulary_targets>\n"
        f"<new_episode_prose>{prose}</new_episode_prose>"
    )
    response = api_call_with_retry(lambda: client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        system=STATE_MANAGER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ))
    if response.stop_reason == "max_tokens":
        print("WARNING: State Manager output truncated.")

    return parse_json_response(response.content[0].text)


# ── Feedback loop ─────────────────────────────────────────────────────────────

def generate_and_classify(episode_plan, confirmed_story_bible, classifier,
                          target_level=TARGET_LEVEL, max_retries=MAX_RETRIES):
    """
    Run the Writer in a loop until its output matches target_level.

    On each failed attempt the classifier verdict is passed back as a
    <classifier_feedback> hint in the next Writer prompt.

    Returns: (prose, classification_result, number_of_attempts)
    """
    feedback = None

    for attempt in range(1, max_retries + 1):
        print(f"\n--- WRITER (attempt {attempt}/{max_retries}) ---")
        if feedback:
            print(f"Feedback to Writer: {feedback}")

        prose  = run_writer(episode_plan, confirmed_story_bible, feedback=feedback)
        result = classify(prose, classifier)
        predicted, confidence = result["level"], result["confidence"]

        print(f"Classifier: predicted={predicted}  confidence={confidence:.2f}  (target={target_level})")
        print(f"All probs: {result['probs']}")

        if should_accept(result, target_level):
            print(f"✓ Accepted on attempt {attempt}")
            return prose, result, attempt

        if attempt < max_retries:
            feedback = build_feedback(result, target_level, prose, classifier)

    print(f"✗ Max retries reached — accepting last output (predicted={predicted}).")
    return prose, result, max_retries


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading classifier...")
    classifier = load_classifier()

    confirmed_story_bible = None
    all_prose = []
    total_episodes = None

    episode = 1
    while True:
        print(f"\n{'='*60}")
        print(f"EPISODE {episode}")
        print(f"{'='*60}")

        print("\n--- DIRECTOR ---")
        episode_plan, story_bible = run_director(story_bible=confirmed_story_bible)

        if confirmed_story_bible is None:
            confirmed_story_bible = story_bible
            total_episodes = confirmed_story_bible.get("series_plan", {}).get("total_episodes", 6)
            print(f"Series plan: {total_episodes} episodes")

        print(f"Episode plan: {json.dumps(episode_plan, indent=2)}")

        prose, result, attempts = generate_and_classify(
            episode_plan, confirmed_story_bible, classifier,
            target_level=TARGET_LEVEL, max_retries=MAX_RETRIES,
        )

        print(f"\n--- ACCEPTED PROSE (after {attempts} attempt(s)) ---")
        print(prose)
        all_prose.append(prose)

        print("\n--- STATE MANAGER ---")
        updated_bible = run_state_manager(prose, confirmed_story_bible, episode_plan.get("vocabulary_targets", []))
        print(json.dumps(updated_bible, indent=2))

        confirmed_story_bible = updated_bible

        if episode >= total_episodes:
            break
        episode += 1

    print(f"\n{'='*60}")
    print(f"COMPLETE STORY ({len(all_prose)} episodes)")
    print(f"{'='*60}")
    for i, ep_prose in enumerate(all_prose, 1):
        print(f"\n--- Episode {i} ---\n")
        print(ep_prose)
