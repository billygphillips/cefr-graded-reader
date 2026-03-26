from dotenv import load_dotenv
import os
import anthropic
from prompts import CEFR_WRITER_SYSTEM_PROMPT

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=400,
    system=CEFR_WRITER_SYSTEM_PROMPT,
    messages=[
        {"role": "user", "content": "Write a short story about a woman who misses her train."}
    ]
)

print(response.content[0].text)
