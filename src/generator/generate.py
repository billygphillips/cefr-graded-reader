from dotenv import load_dotenv
import os
import json
import anthropic
from prompts import DIRECTOR_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SEED = """
A young man gets a weekend job helping to clean out an old, abandoned house in his town.
In a dusty cupboard, he finds an old video camera. The tape inside shows the exact room
he is standing in fifty years ago, and the family is hiding a small metal box under the floorboards.
"""

# Step 1: Director generates the story bible and episode plan
print("--- DIRECTOR ---")
director_response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=6000,
    system=DIRECTOR_SYSTEM_PROMPT,
    messages=[
        {"role": "user", "content": f"<seed>{SEED}</seed>"}
    ]
)

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
director_json = json.loads(cleaned[json_start:json_end])
episode_plan = director_json["episode_plan"]

print(json.dumps(episode_plan, indent=2))

# Step 2: Writer generates prose from the episode plan
print("\n--- WRITER ---")
writer_response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=12000,
    system=WRITER_SYSTEM_PROMPT,
    messages=[
        {"role": "user", "content": f"<episode_plan>{json.dumps(episode_plan, indent=2)}</episode_plan>"}
    ]
)

print(writer_response.content[0].text)
