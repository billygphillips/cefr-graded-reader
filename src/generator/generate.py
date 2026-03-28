from dotenv import load_dotenv
import os
import json
import glob
import datetime
import anthropic
from prompts import DIRECTOR_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT, STATE_MANAGER_SYSTEM_PROMPT

load_dotenv()

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


def get_next_experiment_id():
    pattern = os.path.join(EXPERIMENTS_DIR, "exp_*.json")
    existing = glob.glob(pattern)
    if not existing:
        return "exp_001"
    numbers = []
    for filepath in existing:
        basename = os.path.basename(filepath)  # e.g. "exp_008.json"
        num_str = basename[4:7]               # "008"
        numbers.append(int(num_str))
    next_num = max(numbers) + 1
    return f"exp_{next_num:03d}"


def save_experiment(agent, model, provider, prompt_version, temperature, max_tokens,
                    stop_reason, token_usage, system_prompt, user_message, raw_output, parsed_output=None):
    exp_id = get_next_experiment_id()
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    data = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "agent": agent,
        "model": model,
        "provider": provider,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop_reason": stop_reason,
        "token_usage": token_usage,
        "input": {
            "system_prompt": system_prompt,
            "user_message": user_message,
        },
        "output": {
            "raw": raw_output,
            "parsed": parsed_output or {},
        },
        "observation": "",
        "narrative_quality": None,
        "cefr_compliance": None,
    }

    filepath = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved experiment: {filepath}")
    return exp_id


client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SEED = """
A young man gets a weekend job helping to clean out an old, abandoned house in his town.
In a dusty cupboard, he finds an old video camera. The tape inside shows the exact room
he is standing in fifty years ago, and the family is hiding a small metal box under the floorboards.
"""

# Set to an experiment ID (e.g. "exp_009") to skip the Director and reuse that episode plan.
# Set to None to run the Director fresh.
REUSE_EPISODE_PLAN_FROM = None

# Set to a State Manager experiment ID (e.g. "exp_014") to run Episode 2+ using an existing Story Bible.
# Set to None to run Episode 1 from the seed premise.
STORY_BIBLE_FROM_EXPERIMENT = "exp_014"

# Step 1: Director generates the story bible and episode plan
if REUSE_EPISODE_PLAN_FROM:
    reuse_path = os.path.join(EXPERIMENTS_DIR, f"{REUSE_EPISODE_PLAN_FROM}.json")
    with open(reuse_path) as f:
        reuse_data = json.load(f)
    episode_plan = reuse_data["output"]["parsed"]["episode_plan"]
    story_bible = reuse_data["output"]["parsed"]["story_bible"]
    print(f"--- DIRECTOR SKIPPED: loaded episode plan from {REUSE_EPISODE_PLAN_FROM} ---")
else:
    print("--- DIRECTOR ---")
    if STORY_BIBLE_FROM_EXPERIMENT:
        sb_path = os.path.join(EXPERIMENTS_DIR, f"{STORY_BIBLE_FROM_EXPERIMENT}.json")
        with open(sb_path) as f:
            sb_data = json.load(f)
        # confirmed_story_bible is the canonical state from the last State Manager run.
        # Used for the State Manager input — keeps the Director's speculative summaries out.
        confirmed_story_bible = sb_data['output']['parsed']
        story_bible_for_director = {k: v for k, v in confirmed_story_bible.items() if k != "vocabulary_introduced"}
        director_user_message = f"<story_bible>{json.dumps(story_bible_for_director, indent=2)}</story_bible>"
        print(f"Using story bible from {STORY_BIBLE_FROM_EXPERIMENT}")
    else:
        # Episode 1: Director creates the initial story bible — use it as the confirmed state.
        confirmed_story_bible = None
        director_user_message = f"<seed>{SEED}</seed>"
    try:
        director_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            system=DIRECTOR_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": director_user_message}
            ]
        )
    except anthropic.RateLimitError:
        print("ERROR: Rate limit hit. Wait and retry.")
        raise
    except anthropic.APIError as e:
        print(f"ERROR: API error: {e}")
        raise

    if director_response.stop_reason == "max_tokens":
        print("WARNING: Director output truncated (hit max_tokens limit). Output may be incomplete.")

    director_output = director_response.content[0].text

    # Print Director thinking block if present
    thinking_end = director_output.find("</thinking>")
    if thinking_end != -1:
        thinking_start = director_output.find("<thinking>")
        print(director_output[thinking_start:thinking_end + len("</thinking>")])
        after_thinking = director_output[thinking_end + len("</thinking>"):]
    else:
        after_thinking = director_output

    # Strip markdown code fences, then parse JSON
    cleaned = after_thinking.replace("```json", "").replace("```", "").strip()
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1
    print("=== CLEANED JSON STRING ===")
    print(cleaned[json_start:json_end])
    print("=== END CLEANED JSON ===")
    try:
        director_json = json.loads(cleaned[json_start:json_end])
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse Director JSON: {e}")
        print("Raw output for debugging:")
        print(cleaned[json_start:json_end])
        raise

    for key in ["story_bible", "episode_plan"]:
        if key not in director_json:
            raise KeyError(f"ERROR: Director JSON missing expected key: '{key}'")

    episode_plan = director_json["episode_plan"]
    if confirmed_story_bible is None:
        confirmed_story_bible = director_json["story_bible"]

    save_experiment(
        agent="director",
        model="claude-sonnet-4-6",
        provider="anthropic",
        prompt_version="director_v2",
        temperature=1.0,
        max_tokens=6000,
        stop_reason=director_response.stop_reason,
        token_usage={
            "input_tokens": director_response.usage.input_tokens,
            "output_tokens": director_response.usage.output_tokens,
        },
        system_prompt=DIRECTOR_SYSTEM_PROMPT,
        user_message=director_user_message,
        raw_output=director_output,
        parsed_output=director_json,
    )

print(json.dumps(episode_plan, indent=2))

# Step 2: Writer generates prose from the episode plan
print("\n--- WRITER ---")
try:
    writer_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=12000,
        system=WRITER_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": (
                f"<character_profiles>{json.dumps([{'name': c['name'], 'description': c['description'], 'flaw': c['flaw']} for c in confirmed_story_bible['characters']], indent=2)}</character_profiles>\n"
                f"<episode_plan>{json.dumps(episode_plan, indent=2)}</episode_plan>"
            )}
        ]
    )
except anthropic.RateLimitError:
    print("ERROR: Rate limit hit. Wait and retry.")
    raise
except anthropic.APIError as e:
    print(f"ERROR: API error: {e}")
    raise

if writer_response.stop_reason == "max_tokens":
    print("WARNING: Writer output truncated (hit max_tokens limit). Output may be incomplete.")

writer_output = writer_response.content[0].text

thinking_end = writer_output.find("</thinking>")
if thinking_end != -1:
    writer_prose = writer_output[thinking_end + len("</thinking>"):].strip()
else:
    writer_prose = writer_output

save_experiment(
    agent="writer",
    model="claude-sonnet-4-6",
    provider="anthropic",
    prompt_version="writer_v1",
    temperature=1.0,
    max_tokens=12000,
    stop_reason=writer_response.stop_reason,
    token_usage={
        "input_tokens": writer_response.usage.input_tokens,
        "output_tokens": writer_response.usage.output_tokens,
    },
    system_prompt=WRITER_SYSTEM_PROMPT,
    user_message=(
        f"<character_profiles>{json.dumps([{'name': c['name'], 'description': c['description'], 'flaw': c['flaw']} for c in confirmed_story_bible['characters']], indent=2)}</character_profiles>\n"
        f"<episode_plan>{json.dumps(episode_plan, indent=2)}</episode_plan>"
    ),
    raw_output=writer_output,
    parsed_output={"prose": writer_prose},
)

print(writer_prose)

# Step 3: State Manager updates the Story Bible
print("\n--- STATE MANAGER ---")
state_manager_user_message = (
    f"<current_story_bible>{json.dumps(confirmed_story_bible, indent=2)}</current_story_bible>\n"
    f"<vocabulary_targets>{json.dumps(episode_plan['vocabulary_targets'])}</vocabulary_targets>\n"
    f"<new_episode_prose>{writer_prose}</new_episode_prose>"
)
try:
    state_manager_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        system=STATE_MANAGER_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": state_manager_user_message}
        ]
    )
except anthropic.RateLimitError:
    print("ERROR: Rate limit hit. Wait and retry.")
    raise
except anthropic.APIError as e:
    print(f"ERROR: API error: {e}")
    raise

if state_manager_response.stop_reason == "max_tokens":
    print("WARNING: State Manager output truncated (hit max_tokens limit). Output may be incomplete.")

state_manager_output = state_manager_response.content[0].text

# Strip thinking block
thinking_end = state_manager_output.find("</thinking>")
if thinking_end != -1:
    after_thinking = state_manager_output[thinking_end + len("</thinking>"):]
else:
    after_thinking = state_manager_output

# Strip markdown code fences, then parse JSON
cleaned = after_thinking.replace("```json", "").replace("```", "").strip()
json_start = cleaned.find("{")
json_end = cleaned.rfind("}") + 1
try:
    updated_story_bible = json.loads(cleaned[json_start:json_end])
except json.JSONDecodeError as e:
    print(f"ERROR: Failed to parse State Manager JSON: {e}")
    print("Raw output for debugging:")
    print(cleaned[json_start:json_end])
    raise

save_experiment(
    agent="state_manager",
    model="claude-sonnet-4-6",
    provider="anthropic",
    prompt_version="state_manager_v1",
    temperature=1.0,
    max_tokens=4000,
    stop_reason=state_manager_response.stop_reason,
    token_usage={
        "input_tokens": state_manager_response.usage.input_tokens,
        "output_tokens": state_manager_response.usage.output_tokens,
    },
    system_prompt=STATE_MANAGER_SYSTEM_PROMPT,
    user_message=state_manager_user_message,
    raw_output=state_manager_output,
    parsed_output=updated_story_bible,
)

print(json.dumps(updated_story_bible, indent=2))
