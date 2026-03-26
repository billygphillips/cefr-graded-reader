from dotenv import load_dotenv
import os
import anthropic
from prompts import DIRECTOR_SYSTEM_PROMPT

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SEED = """
A young man gets a weekend job helping to clean out an old, abandoned house in his town.
In a dusty cupboard, he finds an old video camera. The tape inside shows the exact room
he is standing in fifty years ago, and the family is hiding a small metal box under the floorboards.
"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4000,
    system=DIRECTOR_SYSTEM_PROMPT,
    messages=[
        {"role": "user", "content": f"<seed>{SEED}</seed>"}
    ]
)

print(response.content[0].text)
